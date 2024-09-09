import functools
import gc
import json
import math
import os
from collections import Counter
from typing import Callable, Dict, Tuple, Type, Union

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open
from tqdm import tqdm
from transformers.modeling_utils import PretrainedConfig, PreTrainedModel

from morphling._C import tensor_handle
from morphling.common import EmulatorConfig
from morphling.utils import get_checkpoint_paths


class EmulationEngine(object):
    param_id = 0
    request_id = 0
    # request_id_flag = False
    config = {}

    @staticmethod
    def find_tensor_same_size(
        param_meta_map: Dict[str, dict], size: int
    ) -> np.ndarray:
        names_of_size = [
            name for name, param in param_meta_map.items() if param["size"] == size
        ]
        ids_of_size = [param_meta_map[name]["id"] for name in names_of_size]
        ids_of_size = np.array(ids_of_size, dtype=np.uint32)
        return ids_of_size

    @staticmethod
    def compute_shm_offsets(
        param_meta_map: Dict[str, dict]
    ) -> Tuple[int, Dict[str, int]]:
        unique_sizes_counter = Counter(
            [param["size"] for param in param_meta_map.values()]
        )
        shm_mem_size = sum(
            [size + 4 * count for size, count in unique_sizes_counter.items()]
        )

        shm_mem_size_cum = np.cumsum(
            [size + 4 * count for size, count in unique_sizes_counter.items()]
        )
        shm_mem_size_cum = shm_mem_size_cum - shm_mem_size_cum[0]
        shm_mem_offsets = dict(zip(unique_sizes_counter.keys(), shm_mem_size_cum))
        shm_mem_offsets = {k: int(v) for k, v in shm_mem_offsets.items()}

        return shm_mem_size, shm_mem_offsets

    @staticmethod
    def compute_pin_offsets(
        param_meta_map: Dict[str, dict]
    ) -> Tuple[int, Dict[str, int]]:
        pin_mem_size = sum(
            [meta["size"] for _, meta in param_meta_map.items()]
        )
        offset = 0
        pin_mem_offsets = {}
        for name, meta in param_meta_map.items():
            pin_mem_offsets[name] = offset
            offset += meta["size"]
        return pin_mem_size, pin_mem_offsets

    @staticmethod
    def update_shm_offsets(
        param_meta_map: Dict[str, dict]
    ) -> Tuple[int, Dict[str, int]]:
        _, shm_mem_offsets = EmulationEngine.compute_shm_offsets(param_meta_map)
        unique_sizes_counter = Counter(
            [param["size"] for param in param_meta_map.values()]
        )
        for size, count in unique_sizes_counter.items():
            # find all tensor name and id with the same size
            names_of_size = [
                name for name, param in param_meta_map.items() if param["size"] == size
            ]

            for i, name in enumerate(names_of_size):
                param_meta_map[name]["shm_offset"] = shm_mem_offsets[size] + i * 4

        return param_meta_map

    def __init__(self, config: PretrainedConfig):

        self.config = config

        model_name_or_path = config._name_or_path
        if os.path.exists(model_name_or_path):
            checkpoint_paths = get_checkpoint_paths(model_name_or_path)
        else:
            checkpoint_paths = None
            # get the checkpoint download path from huggingface hub
            model_path = snapshot_download(
                model_name_or_path,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
                ignore_patterns=["flax*", "tf*"],
            )
            if model_path is None:
                raise RuntimeError(
                    f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                    f"Please provide a valid checkpoint."
                )
            checkpoint_paths = get_checkpoint_paths(model_path)
        self.ckpt_files = checkpoint_paths

    def init(
        self, cls: Type[PreTrainedModel], config: Union[str, Dict, EmulatorConfig]
    ):

        self.cls = cls
        self.param_meta_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(config, str):
            config = EmulatorConfig.load_from_file(config)
        elif isinstance(config, dict):
            config = EmulatorConfig.load_from_json(config)

        self.checkpoint = config.ckpt_path
        os.makedirs(self.checkpoint, exist_ok=True)

        self.emulator_config = config
        self.tensor_handle = tensor_handle(config.ckpt_path)

        return self

    def __enter__(self):

        def do_nothing_decorator(orig_func: Callable) -> Callable:

            @functools.wraps(orig_func)
            def do_nothing(*args, **kwargs):
                pass

            return do_nothing

        def post_init_decorator(orig_post_init: Callable) -> Callable:
            # FIXME: this is a hacky way to get rid of the write to weight in the post_init, need a better way to do this if we need to support model training
            @functools.wraps(orig_post_init)
            def archer_post_init(cls, *args, **kwargs):
                pass

            return archer_post_init

        def torch_index_select_decorator(orig_torch_index_select: Callable):

            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(input, dim, index.to(input.device)).to(
                    "cuda:0"
                )

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:

            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    param.data = torch.zeros(
                        1, dtype=param.dtype, device=param.device, pin_memory=True
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    buffer.data = torch.zeros(
                        1, dtype=buffer.dtype, device=buffer.device, pin_memory=True
                    )

            return archer_apply_to_model

        def init_decorator(orig_init: Callable) -> Callable:

            @functools.wraps(orig_init)
            def archer_init(cls, config, *args, **kwargs):
                # self.config = config
                pass

            return archer_init

        def param_init_decorator(orig_param_init: Callable) -> Callable:

            @functools.wraps(orig_param_init)
            def archer_param_init(cls, *args, **kwargs):
                orig_param_init(cls, *args, **kwargs)

                cls.param_real_shape = {}
                for name, param in cls.named_parameters(recurse=False):
                    cls.param_real_shape[name] = param.shape
                    param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
                    # self.model_create_counter.update(1)

                for name, buf in cls.named_buffers(recurse=False):
                    cls.param_real_shape[name] = buf.shape
                    buf.data = torch.zeros(1, dtype=buf.dtype, device=buf.device)
                    # self.model_create_counter.update(1)

            return archer_param_init

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = init_decorator(self.cls._old_init)

        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(torch._old_index_select)
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = post_init_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = post_init_decorator(PreTrainedModel._old_post_init)

        # for all the modules in torch.nn, add post_init method
        # assert False, torch.nn.modules.__dict__
        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module._old_init = module.__init__
            module.__init__ = param_init_decorator(module.__init__)

            if hasattr(module, "reset_parameters"):
                module._old_reset_parameters = module.reset_parameters
                module.reset_parameters = do_nothing_decorator(module.reset_parameters)

        def from_pretrained_decorator(orig_from_pretrained: Callable) -> Callable:

            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                # print("Creating model from scratch ...")

                param_meta_map_file = os.path.join(
                    self.checkpoint, "param_meta_map.json"
                )

                self.dtype_cls = self.config.torch_dtype

                if (
                    not self.tensor_handle.is_tensor_index_initialized()
                    or not os.path.exists(param_meta_map_file)
                ):
                    print(
                        "Creating model from scratch ...", self.ckpt_files, flush=True
                    )

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    for ckpt in tqdm(
                        self.ckpt_files, desc="Loading checkpoint files", smoothing=0
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(ckpt, framework="pt", device="cpu") as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        # convert all tensors in state_dict to self.dtype
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.dtype_cls).to("cpu")

                        self._offload_state_dict(state_dict, empty_state_dict)

                        # print("Loading ckpt file", ckpt, flush=True)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    EmulationEngine.update_shm_offsets(self.param_meta_map)

                    with open(param_meta_map_file, "w") as f:
                        json.dump(self.param_meta_map, f)

                else:
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(param_meta_map_file, "r") as f:
                        self.param_meta_map = json.load(f)

                is_flash_attn_available = kwargs.get("is_flash_attn_available", False)

                if self.dtype_cls is torch.bfloat16 or self.dtype_cls is torch.float16:
                    model = cls._from_config(
                        self.config,
                        torch_dtype=self.dtype_cls,
                        attn_implementation=(
                            "flash_attention_2" if is_flash_attn_available else "eager"
                        ),
                    )
                else:
                    model = cls._from_config(self.config)

                base_model_prefix = model.base_model_prefix

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.param_meta_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[(len(base_model_prefix) + 1) :]
                        if name_without_prefix in self.param_meta_map:
                            self.param_meta_map[name] = self.param_meta_map[
                                name_without_prefix
                            ]
                            self.param_meta_map.pop(name_without_prefix)
                    param.ar_id = self.param_meta_map.get(name, None)

                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):

        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module.__init__ = module._old_init

            if hasattr(module, "reset_parameters"):
                module.reset_parameters = module._old_reset_parameters

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            if not param_name in self.param_meta_map:
                param = state_dict[param_name]
                param_id = self._generate_param_id()

                file_offset = self.tensor_handle.offload_tensor(param, param_id)

                self.param_meta_map[param_name] = {
                    "id": param_id,
                    "size": param.numel() * param.element_size(),
                    "shm_offset": -1,
                    "file_offset": file_offset,
                    "shape": tuple(param.shape),
                    "stride": tuple(param.stride()),
                    "dtype": str(param.dtype),
                }

            # if not self.tensor_handle.is_tensor_offloaded(self.param_meta_map[param_name]["id"]):

        gc.collect()
        torch.cuda.empty_cache()
