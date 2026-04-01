import sys

import pytest
import torch

sys.path.append("../../")

from morphling.common.decorators import func_access_decorator
from morphling.common.keywords import *

try:
    from transformers import AutoModelForCausalLM

    _TRANSFORMERS_IMPORT_ERROR = None
except (TypeError, ImportError) as e:
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = e


@pytest.mark.skipif(
    _TRANSFORMERS_IMPORT_ERROR is not None,
    reason=(
        "transformers import is incompatible with current environment: "
        f"{_TRANSFORMERS_IMPORT_ERROR}"
    ),
)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA GPU for training test"
)
@pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="monkey-patching torch APIs in this test is incompatible with Python 3.12",
)
def test_pytorch_train():
    model_name = "facebook/opt-125m"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    inputs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], device="cuda")
    labels = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], device="cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    tensor_functions = [
        f for f in dir(torch.Tensor) if callable(getattr(torch.Tensor, f))
    ]
    for f in tensor_functions:
        if f in pyobj_keywords or f in pytorch_keywords:
            continue
        func = getattr(torch.Tensor, f)
        setattr(torch.Tensor, f, func_access_decorator(func, f))

    torch_functions = [f for f in dir(torch) if callable(getattr(torch, f))]
    for f in torch_functions:
        if f in pyobj_keywords or f in pytorch_keywords:
            continue
        func = getattr(torch, f)
        setattr(torch, f, func_access_decorator(func, f))

    model.train()
    outputs = model(inputs, labels=labels)
    loss = outputs.loss

    assert loss is not None
    loss.backward()
