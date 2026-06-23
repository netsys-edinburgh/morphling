# pyright: reportMissingImports=false, reportAssignmentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportDeprecated=false, reportUnusedImport=false, reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnusedParameter=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportIncompatibleMethodOverride=false, reportArgumentType=false, reportCallIssue=false
"""Autograd hooks for green context switching and GEMM logging."""

import functools
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from morphling.common import get_logger

logger = get_logger()
_backend: Any = None


def set_backend(backend: Any) -> None:
    """Set the active compute backend for morphling hooks.

    Args:
        backend: Backend instance to use for dispatch.
    """
    global _backend
    _backend = backend


_enable_verification = False
_greenctx: Any = None
_gemm_log: List[Dict[str, Any]] = []
_greenctx_t0: float = 0.0
_gemm_idx: int = 0
# Decoupled from _greenctx so backends can request activation without logging
# (or logging without activation). set_greenctx() flips this implicitly to
# preserve historical behavior; callers can override afterwards.
_gemm_log_enabled: bool = False


def set_gemm_logging(enabled: bool) -> None:
    """Enable or disable per-GEMM log entries (independent of greenctx)."""
    global _gemm_log_enabled
    _gemm_log_enabled = bool(enabled)


HOOK_TYPES = [
    "linear",
    "layer_norm",
    "softmax",
    "add",
    "divide",
    "subtract",
    "matmul",
    "relu",
    "gelu",
    "dropout",
]


def _reset_gemm_log() -> None:
    global _gemm_log
    global _greenctx_t0
    global _gemm_idx

    _gemm_log = []
    _greenctx_t0 = 0.0
    _gemm_idx = 0


def _elapsed_us() -> float:
    global _greenctx_t0

    now_s = time.perf_counter()
    if _greenctx_t0 == 0.0:
        _greenctx_t0 = now_s
        return 0.0
    return (now_s - _greenctx_t0) * 1_000_000.0


def _log_gemm(
    gemm_idx: int,
    phase: str,
    start_us: float,
    end_us: float,
    sm_count: Optional[int],
    m: int = 0,
    n: int = 0,
    k: int = 0,
) -> None:
    duration_us = end_us - start_us
    _gemm_log.append(
        {
            "gemm_idx": gemm_idx,
            "phase": phase,
            "start_us": start_us,
            "end_us": end_us,
            "duration_us": duration_us,
            "sm_count": sm_count,
            "greenctx_enabled": _greenctx is not None,
            "m": m,
            "n": n,
            "k": k,
        }
    )
    logger.debug(
        "GEMM log idx=%s phase=%s duration_us=%.3f sm_count=%s",
        gemm_idx,
        phase,
        duration_us,
        sm_count,
    )


def set_greenctx(greenctx: Any = None, reset_log: bool = True) -> None:
    """Set the green context controller and optionally reset the GEMM log.

    Implicitly enables logging when ``greenctx`` is not ``None`` and disables
    it otherwise, preserving the pre-#46 behavior. Call
    :func:`set_gemm_logging` afterwards to override.
    """
    global _greenctx
    global _gemm_log_enabled

    _greenctx = greenctx
    _gemm_log_enabled = greenctx is not None
    if _greenctx is not None and not hasattr(_greenctx, "deactivate"):
        backend = getattr(_greenctx, "backend", None)
        if backend is None:
            backend = getattr(_greenctx, "_backend", None)
        deactivate_fn = getattr(backend, "deactivate", None)
        if callable(deactivate_fn):
            setattr(_greenctx, "deactivate", deactivate_fn)
    if reset_log:
        _reset_gemm_log()


def get_gemm_log(reset: bool = False) -> List[Dict[str, Any]]:
    """Get the GEMM log entries collected during execution."""
    log = list(_gemm_log)
    if reset:
        _reset_gemm_log()
    return log


def apply_hooks(types: Union[str, List[str]], greenctx: Any = None):
    """Apply autograd hooks to PyTorch operations for green context switching."""
    set_greenctx(greenctx)

    if isinstance(types, str):
        types = [types]
    for t in types:
        if t not in HOOK_TYPES:
            raise ValueError(f"Unsupported hook type: {t}")

        if t == "linear":
            torch.nn.functional.linear = LinearFunction.apply
            torch.Tensor.__matmul__ = LinearFunction.apply
            torch.bmm = LinearFunction.apply
            # torch.matmul = LinearFunction.apply

            def forward_decorator(func):
                def wrapper(self, input):
                    return LinearFunction.apply(
                        input, self.weight.t(), self.bias
                    )

                return wrapper

            torch.nn.Linear.forward = forward_decorator(torch.nn.Linear.forward)
            print("Linear hook applied")
        else:
            raise NotImplementedError(f"Hook type {t} is not implemented yet")


# custom autograd function for linear layer
class LinearFunction(torch.autograd.Function):
    extra_dict = {"className": "LinearFunction"}
    logger.debug = functools.partial(logger.debug, extra=extra_dict)

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        global _gemm_idx

        print("LinearFunction forward", input.shape, weight.shape)
        ctx.save_for_backward(input, weight, bias)
        # output = input.mm(weight.t())
        # logger.debug(f"input shape: {input.shape}")
        # logger.debug(f"weight shape: {weight.shape}")
        # output = torch.as_tensor(np.matmul(input, weight))

        start_us: Optional[float] = None
        sm_count: Optional[int] = None
        gemm_idx: Optional[int] = None
        if _greenctx is not None:
            start_us = _elapsed_us()
            activated_sm_count = None
            try:
                sm_count, _ = _greenctx.activate_for_time(int(start_us))
                activated_sm_count = sm_count
                _backend.async_dispatch_matmul(input, weight.transpose(-2, -1))
            finally:
                if activated_sm_count is not None:
                    _greenctx.deactivate(activated_sm_count)
            if _gemm_log_enabled:
                gemm_idx = _gemm_idx
                _gemm_idx += 1
        else:
            _backend.async_dispatch_matmul(input, weight.transpose(-2, -1))
        output = _backend.wait_matmul(0)
        if _gemm_log_enabled and start_us is not None and gemm_idx is not None:
            m = int(input.shape[0])
            k = int(input.shape[1])
            n = int(weight.shape[0])
            _log_gemm(
                gemm_idx,
                "forward",
                start_us,
                _elapsed_us(),
                sm_count,
                m,
                n,
                k,
            )
        # ref = torch.matmul(input.to("cuda:0"), weight.to("cuda:0")).to("cpu")
        # validate output
        # assert torch.allclose(output, ref), f"Output is not close! input shape: {input.shape}, weight shape: {weight.shape}, max diff: {torch.max(torch.abs(output - ref))}"

        if bias is not None:
            output = output + bias.unsqueeze(0).expand_as(output)

        if _enable_verification:
            ref = torch.matmul(input, weight)
            if bias is not None:
                ref = ref + bias.unsqueeze(0).expand_as(ref)
            assert torch.allclose(output, ref, atol=1e-5), (
                f"Output is not close! input shape: {input.shape}, weight shape: {weight.shape}, max diff: {torch.max(torch.abs(output - ref))}"
            )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        global _gemm_idx

        input, weight, bias = ctx.saved_tensors
        print(
            "LinearFunction backward",
            grad_output.shape,
            weight.shape,
            input.shape,
        )
        grad_input = grad_weight = grad_bias = None
        grad_input_start_us: Optional[float] = None
        grad_input_sm_count: Optional[int] = None
        grad_input_gemm_idx: Optional[int] = None
        grad_weight_start_us: Optional[float] = None
        grad_weight_sm_count: Optional[int] = None
        grad_weight_gemm_idx: Optional[int] = None
        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            # grad_input = torch.as_tensor(
            #     np.matmul(grad_output, weight.transpose(-2, -1))
            # )
            # grad_input = _backend.sync_dispatch_matmul(
            #     grad_output, weight.transpose(-2, -1)
            # )
            if _greenctx is not None:
                grad_input_start_us = _elapsed_us()
                activated_sm_count = None
                try:
                    grad_input_sm_count, _ = _greenctx.activate_for_time(
                        int(grad_input_start_us)
                    )
                    activated_sm_count = grad_input_sm_count
                    _backend.async_dispatch_matmul(grad_output, weight)
                finally:
                    if activated_sm_count is not None:
                        _greenctx.deactivate(activated_sm_count)
                if _gemm_log_enabled:
                    grad_input_gemm_idx = _gemm_idx
                    _gemm_idx += 1
            else:
                _backend.async_dispatch_matmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)
            # grad_weight = torch.as_tensor(
            #     np.matmul(grad_output.transpose(-2, -1), input)
            # ).transpose(-2, -1)
            # grad_weight = _backend.sync_dispatch_matmul(
            #     grad_output.transpose(-2, -1), input
            # ).transpose(-2, -1)
            if _greenctx is not None:
                grad_weight_start_us = _elapsed_us()
                activated_sm_count = None
                try:
                    grad_weight_sm_count, _ = _greenctx.activate_for_time(
                        int(grad_weight_start_us)
                    )
                    activated_sm_count = grad_weight_sm_count
                    _backend.async_dispatch_matmul(
                        grad_output.transpose(-2, -1),
                        input.transpose(-2, -1),
                    )
                finally:
                    if activated_sm_count is not None:
                        _greenctx.deactivate(activated_sm_count)
                if _gemm_log_enabled:
                    grad_weight_gemm_idx = _gemm_idx
                    _gemm_idx += 1
            else:
                _backend.async_dispatch_matmul(
                    grad_output.transpose(-2, -1),
                    input.transpose(-2, -1),
                )

        dispatch_count = 0
        if ctx.needs_input_grad[0]:
            grad_input = _backend.wait_matmul(dispatch_count)
            dispatch_count += 1
            if (
                grad_input_start_us is not None
                and grad_input_gemm_idx is not None
            ):
                grad_input_m = int(grad_output.shape[0])
                grad_input_k = int(weight.shape[0])
                grad_input_n = int(weight.shape[1])
                _log_gemm(
                    grad_input_gemm_idx,
                    "backward_grad_input",
                    grad_input_start_us,
                    _elapsed_us(),
                    grad_input_sm_count,
                    grad_input_m,
                    grad_input_n,
                    grad_input_k,
                )

        if ctx.needs_input_grad[1]:
            grad_weight = _backend.wait_matmul(dispatch_count).transpose(-2, -1)
            dispatch_count += 1
            if (
                grad_weight_start_us is not None
                and grad_weight_gemm_idx is not None
            ):
                grad_weight_m = int(weight.shape[0])
                grad_weight_n = int(weight.shape[1])
                grad_weight_k = int(input.shape[0])
                _log_gemm(
                    grad_weight_gemm_idx,
                    "backward_grad_weight",
                    grad_weight_start_us,
                    _elapsed_us(),
                    grad_weight_sm_count,
                    grad_weight_m,
                    grad_weight_n,
                    grad_weight_k,
                )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        if _enable_verification:
            if ctx.needs_input_grad[0]:
                ref_grad_input = torch.matmul(grad_output, weight, atol=1e-5)
                assert torch.allclose(grad_input, ref_grad_input), (
                    f"grad_input is not close! max diff: {torch.max(torch.abs(grad_input - ref_grad_input))}"
                )
            if ctx.needs_input_grad[1]:
                ref_grad_weight = torch.matmul(
                    grad_output.transpose(-2, -1), input
                ).transpose(-2, -1)
                assert torch.allclose(
                    grad_weight, ref_grad_weight, atol=1e-5
                ), (
                    f"grad_weight is not close! max diff: {torch.max(torch.abs(grad_weight - ref_grad_weight))}"
                )
            if bias is not None and ctx.needs_input_grad[2]:
                ref_grad_bias = grad_output.sum(0)
                assert torch.allclose(grad_bias, ref_grad_bias, atol=1e-5), (
                    f"grad_bias is not close! max diff: {torch.max(torch.abs(grad_bias - ref_grad_bias))}"
                )

        return grad_input, grad_weight, grad_bias
