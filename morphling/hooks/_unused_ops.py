"""Autograd Function implementations that are defined but not yet wired via apply_hooks(). Preserved here for potential future use."""

# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedVariable=false, reportImplicitOverride=false, reportIncompatibleMethodOverride=false, reportArgumentType=false, reportCallIssue=false, reportAny=false, reportExplicitAny=false

import numpy as np
import torch

from morphling.common import get_logger

logger = get_logger()


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, normalized_shape, weight, bias, eps):
        logger.debug("LayerNormFunction forward")
        ctx.save_for_backward(input, weight, bias)
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
