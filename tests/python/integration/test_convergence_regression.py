"""Integration convergence regression checks for training stability.

These tests define the red-phase contract for end-to-end training behavior,
mirroring the training loop structure used by the existing greenctx training
integration test.
"""

from typing import Callable, cast

import pytest
import torch

from tests.python.testutils.determinism import set_all_seeds
from tests.python.testutils.numerical import (
    TIER_EXACT,
    assert_no_nan_inf,
    compare_loss_trajectories,
)

pytestmark = pytest.mark.smoke


def _set_torch_seed(seed: int) -> None:
    manual_seed = cast(Callable[[int], torch.Generator], torch.manual_seed)
    _ = manual_seed(seed)


def _make_tiny_model(seed: int) -> torch.nn.Sequential:
    set_all_seeds(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    return model


def _run_convergence_training(
    seed: int = 42, n_steps: int = 20
) -> tuple[list[float], list[float]]:
    model = _make_tiny_model(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    _set_torch_seed(999)
    x = torch.randn(16, 32)
    y = torch.randn(16, 1)

    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        output = cast(torch.Tensor, model(x))
        loss = torch.nn.functional.mse_loss(output, y)
        backward = cast(Callable[[], object], getattr(loss, "backward"))
        _ = backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf")
        )
        losses.append(loss.item())
        grad_norms.append(grad_norm.item())
        step = cast(Callable[[], object], getattr(optimizer, "step"))
        _ = step()

    return losses, grad_norms


@pytest.mark.deep
def test_loss_decreases_over_training() -> None:
    set_all_seeds(42)
    losses, _ = _run_convergence_training(seed=42, n_steps=20)

    assert losses[-1] < losses[0]

    max_consecutive_increases = 0
    consecutive_increases = 0
    for i in range(5, len(losses)):
        if losses[i] > losses[i - 1]:
            consecutive_increases += 1
            max_consecutive_increases = max(
                max_consecutive_increases, consecutive_increases
            )
        else:
            consecutive_increases = 0

    assert max_consecutive_increases <= 3
    assert_no_nan_inf(torch.tensor(losses), name="losses")


@pytest.mark.deep
def test_final_loss_within_golden_tolerance() -> None:
    set_all_seeds(42)
    losses1, _ = _run_convergence_training(seed=42, n_steps=20)

    set_all_seeds(42)
    losses2, _ = _run_convergence_training(seed=42, n_steps=20)

    result = compare_loss_trajectories(losses1, losses2, TIER_EXACT)
    assert result.passed

    assert losses1[-1] / losses1[0] < 0.95


@pytest.mark.deep
def test_no_gradient_explosion() -> None:
    set_all_seeds(42)
    _, grad_norms = _run_convergence_training(seed=42, n_steps=20)

    sorted_norms = sorted(grad_norms)
    median_norm = sorted_norms[len(sorted_norms) // 2]
    assert median_norm > 0.0

    assert all(norm <= 100.0 * median_norm for norm in grad_norms)
    assert all(norm >= 0.001 * median_norm for norm in grad_norms)

    assert_no_nan_inf(torch.tensor(grad_norms), name="grad_norms")
