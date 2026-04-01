"""Axis 3: determinism checks across repeated seeded runs.

These tests define the red-phase contract for seed propagation and result
stability across repeated executions of the same tiny model.
"""

import torch
import pytest


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_same_seed_same_loss_trajectory() -> None:
    """Verify repeated runs with the same seed produce the same loss
    trajectory on the CUDA path.
    """
    pytest.fail("TDD stub: same seed must produce same loss trajectory")


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_same_seed_same_logits() -> None:
    """Verify repeated runs with the same seed produce identical logits on
    the CUDA path.
    """
    pytest.fail("TDD stub: same seed must produce same logits")


@pytest.mark.smoke
def test_different_seed_different_results() -> None:
    """Verify the tiny 2-layer Linear model produces different results when
    seeded differently, using a CPU-only path.
    """
    pytest.fail("TDD stub: different seeds should change results")


@pytest.mark.smoke
def test_seed_propagation_complete() -> None:
    """Verify seed propagation reaches every random source used by the tiny
    2-layer Linear model on the CPU-only path.
    """
    pytest.fail("TDD stub: seed propagation must be complete")
