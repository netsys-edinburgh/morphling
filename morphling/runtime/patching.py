# pyright: reportMissingTypeArgument=false, reportPossiblyUnboundVariable=false, reportAttributeAccessIssue=false, reportOptionalCall=false

"""Patching helpers for model initialization and loading.

Provides:
  - decorators used to bypass expensive/default initialization
  - InitEmptyModel context manager for empty-weight model construction
"""

import functools
from typing import Callable, Type

import torch
from transformers import AutoConfig, PreTrainedModel

MemoryManagerClient = None
set_tensor_shm = None


def do_nothing_decorator(orig_func: Callable) -> Callable:
    """Decorator that replaces a function with a no-op."""

    @functools.wraps(orig_func)
    def do_nothing(*args, **kwargs):
        pass

    return do_nothing


def param_init_decorator(orig_param_init: Callable) -> Callable:
    """Decorator that initializes parameters with empty tensors."""

    @functools.wraps(orig_param_init)
    def empty_param_init(cls, *args, **kwargs):
        orig_param_init(cls, *args, **kwargs)

        for name, param in cls.named_parameters(recurse=False):
            param.data = torch.empty(
                param.shape, dtype=param.dtype, device=param.device
            )

        for name, buf in cls.named_buffers(recurse=False):
            buf.data = torch.empty(
                param.shape, dtype=buf.dtype, device=buf.device
            )

    return empty_param_init


def from_pretrained_decorator(orig_from_pretrained: Callable) -> Callable:
    """Decorator that loads models with shared memory mapping.

    Intercepts from_pretrained to map model parameters to shared memory
    via the MemoryManagerClient.
    """

    @functools.wraps(orig_from_pretrained)
    def archer_from_pretrained(cls, *args, **kwargs):
        config = AutoConfig.from_pretrained(args[0])
        model = cls._from_config(config, torch_dtype=torch.float32)

        print(f"Model config: {config}")
        print(f"Model: {model}")

        client = MemoryManagerClient()
        param_shm_map = client.get_model_param()
        print(f"param_shm_map: {param_shm_map}")

        for name, param in model.named_parameters(recurse=True):
            if name not in param_shm_map:
                print(f"param {name} not found in param_shm_map")
                continue
            shm_name, shm_size = param_shm_map[name]
            tensor = torch.empty(param.data.shape, dtype=param.data.dtype)
            set_tensor_shm(tensor, shm_name, shm_size)
            param.data = tensor
            assert ~(
                torch.isclose(param.data, torch.zeros_like(param.data)).all()
                == True
            ), f"param {name} is zero {param}"

        return model

    return archer_from_pretrained


class InitEmptyModel:
    """Context manager to initialize models with empty tensors.

    Patches torch.nn module initialization to skip actual parameter
    initialization, useful for creating models without loading
    checkpoint weights.

    Usage:
        with InitEmptyModel(OPTForCausalLM):
            model = OPTForCausalLM(config)
    """

    def __init__(self, cls: Type[PreTrainedModel]):
        self.cls = cls

    def __enter__(self):
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

        self.cls._old_post_init = self.cls.post_init
        PreTrainedModel._old_post_init = PreTrainedModel.post_init

        self.cls.post_init = do_nothing_decorator(self.cls.post_init)
        PreTrainedModel.post_init = do_nothing_decorator(
            PreTrainedModel.post_init
        )

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

    def __exit__(self, exc_type, exc_value, traceback):
        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init
        self.cls.from_pretrained = self.cls._old_from_pretrained

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
