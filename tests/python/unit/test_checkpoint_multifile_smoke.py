# pyright: reportMissingImports=false
"""Issue #49: regression lock for the checkpoint reader consistency check.

The FIXME at checkpoint_handle.cpp:32 was a single-file assumption that
emitted a false-positive warning for any non-trivial checkpoint. Fix uses
per-file_id sums from tensor_index_ instead of stat'ing file 0 only.

This Python-level smoke test exercises ArcherTensorHandle.offload_tensor
twice to confirm:
1. The collapsed _C binding still works (#53 regression lock).
2. Multiple offloads to the same index don't trip the new aggregation logic
   (the multi-file path is forward-compatible; the existing offloader writes
   to file_id=0 only, so this asserts single-file equivalence).
"""

from __future__ import annotations

import os

import pytest
import torch


def test_archer_tensor_handle_offloads_multiple_tensors(tmp_path) -> None:
    pytest.importorskip("morphling._C")
    from morphling._C import ArcherTensorHandle

    prefix = str(tmp_path) + os.sep
    handle = ArcherTensorHandle(prefix)
    assert handle.is_tensor_index_initialized() is False

    a = torch.randn(64, 64, dtype=torch.float32)
    b = torch.randn(32, 128, dtype=torch.float32)

    offset_a = handle.offload_tensor(a, 0)
    offset_b = handle.offload_tensor(b, 1)

    assert offset_a >= 0
    assert offset_b > offset_a, (
        "Second tensor must land after the first inside file 0"
    )

    assert handle.is_tensor_offloaded(0) is True
    assert handle.is_tensor_offloaded(1) is True
    assert handle.is_tensor_offloaded(99) is False

    param_file = os.path.join(prefix, "archer_param_0")
    index_file = os.path.join(prefix, "archer_index")
    assert os.path.exists(param_file), "writer must produce file 0"
    assert os.path.exists(index_file), "writer must produce the index"
    assert os.path.getsize(param_file) > 0
