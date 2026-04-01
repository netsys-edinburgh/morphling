"""Axis 2: dispatch correctness checks for block decomposition.

These tests define the red-phase contract for block reassembly, block
placement, and dispatch equivalence on CPU-only paths.
"""

# pyright: reportDeprecated=false

from typing import List, Tuple

import torch
import pytest

from tests.python.testutils.determinism import set_all_seeds
from tests.python.testutils.numerical import (
    TIER_EXACT,
    TIER_STRICT,
    compare_tensors,
)


def _block_matmul_reassemble(
    mat_a: torch.Tensor, mat_b: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, int, List[Tuple[int, int]]]:
    rows, _ = mat_a.shape
    _, cols = mat_b.shape

    out = torch.zeros(rows, cols, dtype=mat_a.dtype)
    block_coords: List[Tuple[int, int]] = []

    for r_start in range(0, rows, block_size):
        r_end = min(r_start + block_size, rows)
        for c_start in range(0, cols, block_size):
            c_end = min(c_start + block_size, cols)
            out[r_start:r_end, c_start:c_end] = (
                mat_a[r_start:r_end, :] @ mat_b[:, c_start:c_end]
            )
            block_coords.append((r_start, c_start))

    return out, len(block_coords), block_coords


@pytest.mark.smoke
def test_block_decomposition_equals_full_matmul() -> None:
    """Verify block-wise decomposition reconstructs the same result as a
    full matrix multiplication on the CPU path.
    """
    set_all_seeds(42)

    block_size = 128
    mat_a = torch.randn(512, 512, dtype=torch.float64)
    mat_b = torch.randn(512, 512, dtype=torch.float64)

    c_ref = torch.matmul(mat_a, mat_b)
    c_reassembled, num_blocks, _ = _block_matmul_reassemble(
        mat_a, mat_b, block_size
    )

    result = compare_tensors(c_reassembled.float(), c_ref.float(), TIER_STRICT)
    assert result.passed
    assert num_blocks == 16


@pytest.mark.smoke
def test_partial_blocks_handled_correctly() -> None:
    """Verify trailing partial blocks are handled correctly during dispatch
    and reassembly without changing the mathematical result.
    """
    set_all_seeds(42)

    block_size = 128
    mat_a = torch.randn(300, 500, dtype=torch.float64)
    mat_b = torch.randn(500, 350, dtype=torch.float64)

    c_ref = torch.matmul(mat_a, mat_b)
    c_reassembled, _, _ = _block_matmul_reassemble(mat_a, mat_b, block_size)

    result = compare_tensors(c_reassembled.float(), c_ref.float(), TIER_STRICT)
    assert result.passed


@pytest.mark.smoke
def test_single_device_equals_no_dispatch() -> None:
    """Verify a single-device configuration behaves like the no-dispatch
    baseline and produces identical outputs.
    """
    set_all_seeds(42)

    block_size = 256
    mat_a = torch.randn(64, 64, dtype=torch.float64)
    mat_b = torch.randn(64, 64, dtype=torch.float64)

    c_ref = torch.matmul(mat_a, mat_b)
    c_dispatched, num_blocks, _ = _block_matmul_reassemble(
        mat_a, mat_b, block_size
    )

    assert num_blocks == 1
    result = compare_tensors(c_dispatched, c_ref, TIER_EXACT)
    assert result.passed


@pytest.mark.smoke
def test_wrong_block_placement_detected() -> None:
    """Intentionally swaps two block positions in reassembly, asserts
    comparison fails.
    """
    set_all_seeds(42)

    block_size = 128
    mat_a = torch.randn(256, 256, dtype=torch.float64)
    mat_b = torch.randn(256, 256, dtype=torch.float64)

    c_ref = torch.matmul(mat_a, mat_b)
    c_correct, _, _ = _block_matmul_reassemble(mat_a, mat_b, block_size)

    c_wrong = c_correct.clone()
    c_wrong[0:128, 0:128] = c_correct[128:256, 128:256]
    c_wrong[128:256, 128:256] = c_correct[0:128, 0:128]

    result = compare_tensors(c_wrong, c_ref, TIER_STRICT)
    assert not result.passed
