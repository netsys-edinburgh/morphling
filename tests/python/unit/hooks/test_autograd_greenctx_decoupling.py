# pyright: reportMissingImports=false, reportAssignmentType=false
"""Tests for the decoupling of greenctx activation from GEMM logging (#46).

The four cases mirror the plan:
- Test A: set_greenctx(fake) implicitly enables logging.
- Test B: set_greenctx(fake) + set_gemm_logging(False) keeps activation, drops logging.
- Test C: set_greenctx(None) disables both activation and logging; dispatch still runs.
- Test D: backward path equivalent of A.
"""

from __future__ import annotations

import pytest
import torch

from morphling.hooks import autograd as ag


class _StubBackend:
    """Counts dispatches and returns a deterministic matmul result."""

    def __init__(self) -> None:
        self.dispatch_calls = 0
        self.wait_calls = 0
        self._queue: list[torch.Tensor] = []

    def async_dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> None:
        self.dispatch_calls += 1
        self._queue.append(torch.matmul(mat_a, mat_b.transpose(-2, -1)))

    def wait_matmul(self, _idx: int) -> torch.Tensor:
        self.wait_calls += 1
        return self._queue.pop(0)


class _StubGreenCtx:
    """Records activate / deactivate calls without doing any real partitioning."""

    def __init__(self) -> None:
        self.activate_calls: list[int] = []
        self.deactivate_calls: list[int] = []

    def activate_for_time(self, t_us: int) -> tuple[int, int]:
        self.activate_calls.append(t_us)
        return (16, 0)

    def deactivate(self, sm_count: int) -> None:
        self.deactivate_calls.append(sm_count)


@pytest.fixture(autouse=True)
def _isolate_module_state():
    """Reset autograd module globals around each test."""
    prev_backend = ag._backend
    prev_greenctx = ag._greenctx
    prev_log_enabled = getattr(ag, "_gemm_log_enabled", False)
    ag._reset_gemm_log()
    yield
    ag._backend = prev_backend
    ag._greenctx = prev_greenctx
    if hasattr(ag, "_gemm_log_enabled"):
        ag._gemm_log_enabled = prev_log_enabled
    ag._reset_gemm_log()


def _run_forward(backend: _StubBackend) -> torch.Tensor:
    ag.set_backend(backend)
    inp = torch.randn(4, 8)
    weight = torch.randn(
        8, 2
    )  # forward.transpose(-2,-1) before backend dispatch
    return ag.LinearFunction.apply(inp, weight, None)


def test_a_greenctx_set_enables_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _StubGreenCtx()
    backend = _StubBackend()
    ag.set_greenctx(ctx)

    _run_forward(backend)

    log = ag.get_gemm_log()
    assert len(log) == 1, "set_greenctx(non-None) must enable logging"
    assert ctx.activate_calls and ctx.deactivate_calls, (
        "activation path must run"
    )
    assert backend.dispatch_calls == 1
    assert backend.wait_calls == 1


def test_b_explicit_logging_off_keeps_activation() -> None:
    ctx = _StubGreenCtx()
    backend = _StubBackend()
    ag.set_greenctx(ctx)
    ag.set_gemm_logging(False)

    _run_forward(backend)

    log = ag.get_gemm_log()
    assert log == [], "set_gemm_logging(False) must suppress logging"
    assert ctx.activate_calls and ctx.deactivate_calls, (
        "activation must still run"
    )
    assert backend.dispatch_calls == 1


def test_c_no_greenctx_disables_both() -> None:
    backend = _StubBackend()
    ag.set_greenctx(None)

    _run_forward(backend)

    log = ag.get_gemm_log()
    assert log == [], "set_greenctx(None) must disable logging"
    assert backend.dispatch_calls == 1, "matmul dispatch must still happen"


def test_d_backward_logs_when_greenctx_set() -> None:
    ctx = _StubGreenCtx()
    backend = _StubBackend()
    ag.set_greenctx(ctx)

    inp = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 2, requires_grad=True)
    ag.set_backend(backend)
    out = ag.LinearFunction.apply(inp, weight, None)
    out.sum().backward()

    log = ag.get_gemm_log()
    phases = {entry["phase"] for entry in log}
    assert "forward" in phases
    assert "backward_grad_input" in phases or "backward_grad_weight" in phases
    assert len(ctx.activate_calls) >= 2
    assert len(ctx.deactivate_calls) >= 2
