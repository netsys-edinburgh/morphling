# pyright: reportMissingImports=false
"""Issue #53: morphling._C must export ArcherTensorHandle and set_tensor_shm.

The plan splits #53 into:
- Track A: bind ArcherTensorHandle (it exists in csrc/, just was not registered).
- Track B: MemoryManagerClient is intentionally absent; the dependent code path
  must error clearly, not at import time.
"""

from __future__ import annotations

import os
import tempfile

import pytest


def test_c_module_exports_archer_tensor_handle() -> None:
    import morphling._C as _c

    assert hasattr(_c, "ArcherTensorHandle"), (
        "ArcherTensorHandle must be exported from morphling._C (#53 Track A)"
    )
    assert hasattr(_c, "set_tensor_shm"), (
        "set_tensor_shm regression check (#53 Track A)"
    )


def test_archer_tensor_handle_constructs_on_empty_prefix() -> None:
    from morphling._C import ArcherTensorHandle

    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "ckpt-")
        handle = ArcherTensorHandle(prefix)
        assert handle.is_tensor_index_initialized() is False


def test_memory_manager_client_absence_is_handled() -> None:
    """Track B: importing model_emulator must not crash even though
    MemoryManagerClient is intentionally missing."""
    pytest.importorskip("transformers")
    import morphling.runtime.model_emulator as me

    assert hasattr(me, "EmulationEngine")
    assert hasattr(me, "MemoryManagerClient"), (
        "model_emulator must expose a guarded MemoryManagerClient name (may be None)"
    )
