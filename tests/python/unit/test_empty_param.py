import sys

import pytest
import torch

try:
    from transformers import AutoConfig, AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except (TypeError, ImportError) as e:
    AutoConfig = None
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = e


def torch_decorator(func, funcname=None):
    def wrapper(*args, **kwargs):
        print("torch_decorator ", funcname)
        return func(*args, **kwargs)

    return wrapper


pyobj_keywords = [
    "__class__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__getattribute__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
    "Tensor",
]


@pytest.mark.skipif(
    _TRANSFORMERS_IMPORT_ERROR is not None,
    reason=(
        "transformers import is incompatible with current environment: "
        f"{_TRANSFORMERS_IMPORT_ERROR}"
    ),
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="global torch monkey-patching in this test is incompatible with Python 3.12",
)
def test_empty_param_generate():
    a = torch.ones((1, 3))
    b = torch.ones((1, 3))
    assert a.shape == b.shape

    model_name = "facebook/opt-1.3b"
    config = AutoConfig.from_pretrained(model_name)
    config.hidden_size = 1
    config.num_attention_heads = 1
    config.word_embed_proj_dim = 1
    model = AutoModelForCausalLM.from_config(config)

    for name, param in model.named_parameters():
        if "embed" in name:
            continue

        shape = [1 for _ in param.shape]
        param.data = torch.rand(shape)

    tensor_functions = [
        f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))
    ]
    for f in tensor_functions:
        if f in pyobj_keywords:
            continue
        func = getattr(torch.Tensor, f)
        setattr(torch.Tensor, f, torch_decorator(func, f))

    torch_functions = [f for f in dir(torch) if callable(getattr(torch, f))]
    for f in torch_functions:
        if f in pyobj_keywords:
            continue
        func = getattr(torch, f)
        setattr(torch, f, torch_decorator(func, f))

    inputs = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(
        inputs, do_sample=True, max_length=50, pad_token_id=50256
    )

    assert output is not None
