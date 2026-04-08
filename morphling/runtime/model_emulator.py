# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportUninitializedInstanceVariable=false, reportAttributeAccessIssue=false, reportPossiblyUnboundVariable=false

import functools
import gc
import json
import os
from typing import Callable, Dict, Type, Union

import torch
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from morphling._C import ArcherTensorHandle, MemoryManagerClient, set_tensor_shm
from morphling.common import EmulatorConfig
import morphling.runtime.patching as _patching
import morphling.runtime.shm_mapping as _shm_mapping
from morphling.runtime.checkpoint_loader import (
    discover_checkpoints,
    offload_state_dict,
)
from morphling.runtime.patching import (
    InitEmptyModel,
    do_nothing_decorator,
    from_pretrained_decorator,
    param_init_decorator,
)
from morphling.runtime.shm_mapping import map_params_to_shm

_patching.MemoryManagerClient = MemoryManagerClient
_patching.set_tensor_shm = set_tensor_shm
_shm_mapping.set_tensor_shm = set_tensor_shm


class EmulationEngine(object):
    param_id = 0
    request_id = 0
    config = {}

    def __init__(self, config: PretrainedConfig):
        self.config = config
        self.ckpt_files = discover_checkpoints(config._name_or_path)

    def init(
        self,
        cls: Type[PreTrainedModel],
        config: Union[str, Dict, EmulatorConfig],
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
        self.ArcherTensorHandle = ArcherTensorHandle(config.ckpt_path)
        return self

    def __enter__(self):
        def post_init_decorator(orig_post_init: Callable) -> Callable:
            @functools.wraps(orig_post_init)
            def archer_post_init(cls, *args, **kwargs):
                pass

            return archer_post_init

        def torch_index_select_decorator(orig_torch_index_select: Callable):
            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(
                    input, dim, index.to(input.device)
                ).to("cuda:0")

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:
            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    param.data = torch.zeros(
                        1,
                        dtype=param.dtype,
                        device=param.device,
                        pin_memory=True,
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.param_meta_map:
                        continue
                    buffer.data = torch.zeros(
                        1,
                        dtype=buffer.dtype,
                        device=buffer.device,
                        pin_memory=True,
                    )

            return archer_apply_to_model

        def init_decorator(orig_init: Callable) -> Callable:
            @functools.wraps(orig_init)
            def archer_init(cls, config, *args, **kwargs):
                pass

            return archer_init

        def param_init_decorator(orig_param_init: Callable) -> Callable:
            @functools.wraps(orig_param_init)
            def archer_param_init(cls, *args, **kwargs):
                orig_param_init(cls, *args, **kwargs)

                cls.param_real_shape = {}
                for name, param in cls.named_parameters(recurse=False):
                    cls.param_real_shape[name] = param.shape
                    param.data = torch.empty(
                        param.shape, dtype=param.dtype, device=param.device
                    )

                for name, buf in cls.named_buffers(recurse=False):
                    cls.param_real_shape[name] = buf.shape
                    buf.data = torch.empty(
                        param.shape, dtype=buf.dtype, device=buf.device
                    )

            return archer_param_init

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = init_decorator(self.cls._old_init)

        torch.nn.modules.module.Module._old_apply = (
            torch.nn.modules.module.Module.apply
        )
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select
        )
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = post_init_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = post_init_decorator(
            PreTrainedModel._old_post_init
        )

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
                module.reset_parameters = do_nothing_decorator(
                    module.reset_parameters
                )

        def from_pretrained_decorator(
            orig_from_pretrained: Callable,
        ) -> Callable:
            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                param_meta_map_file = os.path.join(
                    self.checkpoint, "param_meta_map.json"
                )
                self.dtype_cls = self.config.torch_dtype

                if (
                    not self.ArcherTensorHandle.is_tensor_index_initialized()
                    or not os.path.exists(param_meta_map_file)
                ):
                    print(
                        "Creating model from scratch ...",
                        self.ckpt_files,
                        flush=True,
                    )
                    self.cls.__init__ = self.cls._old_init
                    self.param_meta_map = offload_state_dict(
                        self.ckpt_files,
                        self.ArcherTensorHandle,
                        self.dtype_cls,
                    )
                    if self.param_meta_map:
                        self.param_id = (
                            max(
                                meta["id"]
                                for meta in self.param_meta_map.values()
                            )
                            + 1
                        )
                    with open(param_meta_map_file, "w") as f:
                        json.dump(self.param_meta_map, f)
                else:
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    with open(param_meta_map_file, "r") as f:
                        self.param_meta_map = json.load(f)

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False
                )
                if (
                    self.dtype_cls is torch.bfloat16
                    or self.dtype_cls is torch.float16
                ):
                    model = cls._from_config(
                        self.config,
                        torch_dtype=self.dtype_cls,
                        attn_implementation=(
                            "flash_attention_2"
                            if is_flash_attn_available
                            else "eager"
                        ),
                    )
                else:
                    model = cls._from_config(self.config)

                base_model_prefix = model.base_model_prefix
                for name, param in model.named_parameters(recurse=True):
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[
                            (len(base_model_prefix) + 1) :
                        ]
                        if name_without_prefix in self.param_meta_map:
                            self.param_meta_map[name] = self.param_meta_map[
                                name_without_prefix
                            ]
                            self.param_meta_map.pop(name_without_prefix)
                    param.ar_id = self.param_meta_map.get(name, None)

                self.client = MemoryManagerClient()
                param_shm_map = self.client.get_model_param()
                print(f"param_shm_map: {param_shm_map}")
                return map_params_to_shm(model, param_shm_map, self.client)

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = (
            torch.nn.modules.module.Module._old_apply
        )
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
            if param_name not in self.param_meta_map:
                param = state_dict[param_name]
                param_id = self._generate_param_id()
                file_offset = self.ArcherTensorHandle.offload_tensor(
                    param, param_id
                )

                self.param_meta_map[param_name] = {
                    "id": param_id,
                    "size": param.numel() * param.element_size(),
                    "file_offset": file_offset,
                    "shape": tuple(param.shape),
                    "stride": tuple(param.stride()),
                    "dtype": str(param.dtype),
                }

        gc.collect()
        torch.cuda.empty_cache()
