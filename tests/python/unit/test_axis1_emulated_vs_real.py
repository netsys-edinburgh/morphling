"""Axis 1: emulated vs real numerical consistency checks.

These tests define the red-phase contract for comparing emulator outputs
against golden references and detecting intentional mismatches.
"""

import torch
import pytest


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_logits_match_golden_reference() -> None:
    """Verify emulator logits match the golden reference within the
    comparison tolerances used by compare_logits (1e-3, 1e-4, 1e-5).
    """
    pytest.fail("TDD stub: compare logits against golden reference")


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_loss_trajectory_within_tolerance() -> None:
    """Verify the training loss trajectory stays within the expected
    tolerance window relative to the golden reference run.
    """
    pytest.fail("TDD stub: compare loss trajectory against golden tolerance")


@pytest.mark.deep
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grad_norms_within_range() -> None:
    """Verify gradient norms remain within the acceptable numerical range
    when comparing emulator execution to the real baseline.
    """
    pytest.fail("TDD stub: compare gradient norms against expected range")


@pytest.mark.smoke
def test_intentionally_wrong_logits_detected() -> None:
    """Loads golden logits, corrupts them by +1.0, asserts comparison FAILS —
    validates the comparison utility catches real problems.
    """
    pytest.fail("TDD stub: intentionally wrong logits must be detected")
