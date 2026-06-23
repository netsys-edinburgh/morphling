# pyright: reportMissingImports=false
"""Issue #53: morphling._C must export ArcherTensorHandle and set_tensor_shm.

Track A: bind ArcherTensorHandle (it exists in csrc/, just was not registered).

The former Track B case covered MemoryManagerClient's intentional absence via
the now-removed model_emulator path (deleted in #57); only the live binding
regression checks remain.
"""

from __future__ import annotations

import os
import tempfile


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
