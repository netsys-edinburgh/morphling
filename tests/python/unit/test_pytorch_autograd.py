import pytest
import torch

from morphling.hooks.autograd import (
    AddFunction,
    LayerNormFunction,
    LinearFunction,
)

try:
    from transformers import AutoConfig
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
except (ImportError, TypeError):
    pytest.skip(
        "transformers unavailable or incompatible", allow_module_level=True
    )


def test_autograd_hooks_forward_backward():
    model_name = "facebook/opt-125m"
    config = AutoConfig.from_pretrained(model_name)
    layer = OPTDecoderLayer(config)

    orig_bmm = torch.bmm
    orig_add = torch.Tensor.__add__
    orig_ln = torch.nn.functional.layer_norm

    try:
        torch.bmm = LinearFunction.apply

        def add_arguments(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        torch.Tensor.__add__ = add_arguments(AddFunction.apply)
        torch.nn.functional.layer_norm = LayerNormFunction.apply

        inputs = torch.randn(1, 1, 768)
        output = layer(inputs)
        hidden_states = output[0]

        labels = torch.randn(hidden_states.shape)
        loss = torch.nn.functional.mse_loss(hidden_states, labels)
        loss.backward()
    finally:
        torch.bmm = orig_bmm
        torch.Tensor.__add__ = orig_add
        torch.nn.functional.layer_norm = orig_ln
