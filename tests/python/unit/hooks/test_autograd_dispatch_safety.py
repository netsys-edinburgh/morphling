from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from morphling.hooks import autograd as ag


class _SequenceBackend:
    def __init__(self, operation_ids: list[int]) -> None:
        self._operation_ids = iter(operation_ids)
        self._outputs: dict[int, torch.Tensor] = {}
        self.dispatch_operands: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.waited_ids: list[int] = []

    def async_dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> int:
        oid = next(self._operation_ids)
        self.dispatch_operands.append((mat_a, mat_b))
        if oid >= 0:
            self._outputs[oid] = torch.matmul(
                mat_a, mat_b.transpose(-2, -1)
            )
        return oid

    def wait_matmul(self, oid: int) -> torch.Tensor:
        self.waited_ids.append(oid)
        return self._outputs.pop(oid)


@pytest.fixture(autouse=True)
def _isolate_autograd_state() -> Iterator[None]:
    previous_backend = ag._backend
    previous_greenctx = ag._greenctx
    ag.set_greenctx(None)
    yield
    ag._backend = previous_backend
    ag._greenctx = previous_greenctx


def test_forward_rejects_failed_dispatch_before_wait() -> None:
    backend = _SequenceBackend([-1])
    ag.set_backend(backend)
    input_tensor = torch.randn(3, 4)
    weight = torch.randn(2, 4)

    with pytest.raises(RuntimeError, match="forward.*-1"):
        ag.LinearFunction.apply(input_tensor, weight.transpose(-2, -1), None)

    assert backend.waited_ids == []


def test_backward_rejects_failed_dispatch_before_wait() -> None:
    backend = _SequenceBackend([0, -1, 1])
    ag.set_backend(backend)
    input_tensor = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(2, 4, requires_grad=True)
    output = ag.LinearFunction.apply(
        input_tensor, weight.transpose(-2, -1), None
    )

    with pytest.raises(RuntimeError, match="backward_grad_input.*-1"):
        output.sum().backward()

    assert backend.waited_ids == [0]


def test_forward_dispatches_contiguous_operands_with_native_linear_parity() -> None:
    backend = _SequenceBackend([7])
    ag.set_backend(backend)
    input_tensor = torch.randn(3, 4)
    weight = torch.randn(4, 2)
    bias = torch.randn(2)
    expected = torch.nn.functional.linear(
        input_tensor, weight.transpose(-2, -1), bias
    )

    actual = ag.LinearFunction.apply(input_tensor, weight, bias)

    mat_a, mat_b = backend.dispatch_operands[0]
    assert mat_a.is_contiguous()
    assert mat_b.is_contiguous()
    assert torch.allclose(actual, expected)
