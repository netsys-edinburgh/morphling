"""Axis 2: dispatch correctness checks for block decomposition.

These tests define the red-phase contract for block reassembly, block
placement, and dispatch equivalence on CPU-only paths.
"""

import torch
import pytest

del torch


@pytest.mark.smoke
def test_block_decomposition_equals_full_matmul() -> None:
    """Verify block-wise decomposition reconstructs the same result as a
    full matrix multiplication on the CPU path.
    """
    pytest.fail("TDD stub: block decomposition must match full matmul")


@pytest.mark.smoke
def test_partial_blocks_handled_correctly() -> None:
    """Verify trailing partial blocks are handled correctly during dispatch
    and reassembly without changing the mathematical result.
    """
    pytest.fail("TDD stub: partial blocks must be handled correctly")


@pytest.mark.smoke
def test_single_device_equals_no_dispatch() -> None:
    """Verify a single-device configuration behaves like the no-dispatch
    baseline and produces identical outputs.
    """
    pytest.fail("TDD stub: single-device dispatch must equal baseline")


@pytest.mark.smoke
def test_wrong_block_placement_detected() -> None:
    """Intentionally swaps two block positions in reassembly, asserts
    comparison fails.
    """
    pytest.fail("TDD stub: wrong block placement must be detected")
