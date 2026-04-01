"""Axis 3: determinism checks across repeated seeded runs.

These tests define the red-phase contract for seed propagation and result
stability across repeated executions of the same tiny model.
"""

from typing import Callable, cast

import pytest
import torch

from tests.python.testutils.determinism import (
    deterministic_context,
    set_all_seeds,
)
from tests.python.testutils.numerical import (
    TIER_EXACT,
    compare_loss_trajectories,
    compare_tensors,
)


def _set_torch_seed(seed: int) -> None:
    manual_seed = cast(Callable[[int], torch.Generator], torch.manual_seed)
    _ = manual_seed(seed)


def _tiny_training_loop(seed: int, n_steps: int = 10) -> list[float]:
    set_all_seeds(seed)
    with deterministic_context():
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )

    optimizer: torch.optim.Optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01
    )

    _set_torch_seed(999)
    fixed_input = torch.randn(4, 32)
    target = torch.zeros(4, 1)

    losses: list[float] = []
    for _ in range(n_steps):
        set_all_seeds(seed)
        optimizer.zero_grad(set_to_none=True)
        with deterministic_context():
            output = cast(torch.Tensor, model(fixed_input))
            loss = torch.nn.functional.mse_loss(output, target)
        backward = cast(Callable[[], object], getattr(loss, "backward"))
        _ = backward()
        step = cast(Callable[[], object], getattr(optimizer, "step"))
        _ = step()
        losses.append(float(loss.item()))

    return losses


@pytest.mark.deep
def test_same_seed_same_loss_trajectory() -> None:
    losses1 = _tiny_training_loop(seed=42)
    losses2 = _tiny_training_loop(seed=42)

    result = compare_loss_trajectories(losses1, losses2, TIER_EXACT)
    assert result.passed


@pytest.mark.deep
def test_same_seed_same_logits() -> None:
    set_all_seeds(42)
    with deterministic_context():
        model1 = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
    _ = model1.eval()

    set_all_seeds(42)
    with deterministic_context():
        model2 = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
    _ = model2.eval()

    with torch.no_grad():
        _set_torch_seed(42)
        x1 = torch.randn(4, 32)
        with deterministic_context():
            out1 = cast(torch.Tensor, model1(x1))

        _set_torch_seed(42)
        x2 = torch.randn(4, 32)
        with deterministic_context():
            out2 = cast(torch.Tensor, model2(x2))

    result = compare_tensors(out1, out2, TIER_EXACT)
    assert result.passed


@pytest.mark.smoke
def test_different_seed_different_results() -> None:
    losses1 = _tiny_training_loop(seed=42)
    losses2 = _tiny_training_loop(seed=43)

    assert losses1 != losses2
    assert losses1[0] != losses2[0]


@pytest.mark.smoke
def test_seed_propagation_complete() -> None:
    set_all_seeds(42)
    with deterministic_context():
        assert torch.initial_seed() == 42
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

        model = torch.nn.Sequential(
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        )
        optimizer: torch.optim.Optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01
        )

        x = torch.randn(4, 32)
        y = torch.randn(4, 1)
        pred = cast(torch.Tensor, model(x))
        loss = torch.nn.functional.mse_loss(pred, y)
        backward = cast(Callable[[], object], getattr(loss, "backward"))
        _ = backward()
        step = cast(Callable[[], object], getattr(optimizer, "step"))
        _ = step()
