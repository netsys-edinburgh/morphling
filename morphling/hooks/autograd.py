import torch
import numpy as np
import functools
from morphling.common import get_logger

logger = get_logger()


# custom autograd function for linear layer
class LinearFunction(torch.autograd.Function):
    extra_dict = {"className": "LinearFunction"}
    logger.debug = functools.partial(logger.debug, extra=extra_dict)

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        logger.debug("LinearFunction forward")
        ctx.save_for_backward(input, weight, bias)
        # output = input.mm(weight.t())
        logger.debug(f"input shape: {input.shape}")
        logger.debug(f"weight shape: {weight.shape}")
        output = torch.as_tensor(np.matmul(input, weight))
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("LinearFunction backward")
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        logger.debug(f"input shape: {input.shape}")
        logger.debug(f"grad_output shape: {grad_output.shape}")
        logger.debug(f"weight shape: {weight.shape}")
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = torch.as_tensor(np.matmul(grad_output, weight.transpose(-2,-1)))
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            grad_weight = torch.as_tensor(np.matmul(grad_output.transpose(-2,-1), input)).transpose(-2,-1)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


# custom autograd function for torch.nn.functional.layer_norm
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        logger.debug("LayerNormFunction forward")
        ctx.save_for_backward(input, weight, bias)
        # output = torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)
        output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("LayerNormFunction backward")
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output
        if ctx.needs_input_grad[3]:
            grad_bias = grad_output

        return grad_input, None, grad_weight, grad_bias, None


# custom autograd function for softmax
class SoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.softmax(input, dim=1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        return grad_input


# custom autograd function for torch.Tensor.__add__(self, other) -> Tensor
class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        logger.debug("AddFunction forward")
        output = torch.as_tensor(np.add(input, other))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("AddFunction backward")
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_other = grad_output

        return grad_input, grad_other

# custom autograd function for divide
class DivideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input / other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output / other
        if ctx.needs_input_grad[1]:
            grad_other = -grad_output * input / (other**2)

        return grad_input, grad_other


# custom autograd function for subtract
class SubtractFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input - other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            grad_other = -grad_output

        return grad_input, grad_other


# custom autograd function for matmul
class MatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_for_backward(input, other)
        output = input @ other
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        grad_input = grad_other = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ other.t()
        if ctx.needs_input_grad[1]:
            grad_other = input.t() @ grad_output

        return grad_input, grad_other


# custom autograd function for relu
class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.relu(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0

        return grad_input


# custom autograd function for gelu
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.gelu(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output * torch.nn.functional.gelu(input, True)

        return grad_input


# custom autograd function for dropout
class DropoutFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, p, train):
        ctx.save_for_backward(input)
        if train:
            output = torch.nn.functional.dropout(input, p)
        else:
            output = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output

        return grad_input
