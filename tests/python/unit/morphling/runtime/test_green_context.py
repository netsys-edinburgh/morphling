"""Unit tests for green context Python layer.

Tests backend selection, StreamBundle, step_scope, and
trace-driven SM switching. Requires CUDA GPU with green
context support (CC 8.x + driver 12.5+).

Run:
    pytest tests/python/unit/morphling/runtime/test_green_context.py -v
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# ── Bootstrap: stub morphling to avoid _Msg/_C imports ────


def _bootstrap_morphling():
    """Stub morphling top-level to avoid _Msg/_C deps."""
    root = Path(__file__).resolve().parents[5]  # tests/python/unit/morphling/runtime -> project root
    sys.path.insert(0, str(root))

    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(root / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [
            str(root / "morphling" / "runtime")
        ]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod

    # Load _GreenCtx .so if present
    so_path = root / "morphling" / "_GreenCtx.so"
    if (
        so_path.exists()
        and "morphling._GreenCtx" not in sys.modules
    ):
        spec = importlib.util.spec_from_file_location(
            "morphling._GreenCtx", str(so_path)
        )
        _gc = importlib.util.module_from_spec(spec)
        sys.modules["morphling._GreenCtx"] = _gc
        spec.loader.exec_module(_gc)


_bootstrap_morphling()

from morphling.runtime.green_context import (
    GreenContextConfig,
    GreenContextController,
)
from morphling.runtime.green_context_backends import (
    CppBackend,
    OffBackend,
    StreamBundle,
    select_backend,
)


# ── Fixtures ──────────────────────────────────────────────


def _get_idle_gpu() -> int:
    """Return the ID of an idle GPU (prefer 7, 6, 5)."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPUs available")
    for gid in [7, 6, 5]:
        if gid < torch.cuda.device_count():
            return gid
    return 0


def _greenctx_supported(gpu_id: int) -> bool:
    """Check if green contexts work on this GPU."""
    try:
        b = CppBackend(gpu_id=gpu_id)
        ok = b.is_supported()
        b.close()
        return ok
    except Exception:
        return False


@pytest.fixture(scope="module")
def gpu_id():
    gid = _get_idle_gpu()
    if not _greenctx_supported(gid):
        pytest.skip(
            f"Green contexts not supported on GPU {gid}"
        )
    return gid


@pytest.fixture
def trace_path():
    root = Path(__file__).resolve().parents[5]
    p = root / "tests" / "data" / "greenctx" / "second_level.trace"
    if not p.exists():
        pytest.skip("Trace file not found")
    return str(p)


# ── OffBackend tests ──────────────────────────────────────


class TestOffBackend:
    def test_not_supported(self):
        off = OffBackend(gpu_id=0)
        assert not off.is_supported()
        off.close()

    def test_bundle_has_streams(self):
        off = OffBackend(gpu_id=0)
        b = off.get_stream_bundle(0)
        assert isinstance(b.comp, torch.cuda.Stream)
        assert isinstance(b.recv, torch.cuda.Stream)
        assert isinstance(b.send, torch.cuda.Stream)
        assert isinstance(b.dp, torch.cuda.Stream)
        assert b.sm_count == 0
        off.close()

    def test_strict_raises(self):
        with pytest.raises(RuntimeError, match="strict"):
            OffBackend(gpu_id=0, strict=True)

    def test_activate_noop(self):
        off = OffBackend(gpu_id=0)
        sm, gen = off.activate_for_step(42)
        assert sm == 0
        assert gen == 0
        off.deactivate(0)
        off.close()


# ── CppBackend tests ─────────────────────────────────────


class TestCppBackend:
    def test_supported(self, gpu_id):
        cpp = CppBackend(gpu_id=gpu_id)
        assert cpp.is_supported()
        cpp.close()

    def test_sm_step(self, gpu_id):
        cpp = CppBackend(gpu_id=gpu_id)
        assert cpp.sm_step() == 2  # RTX A5000 CC 8.6
        cpp.close()

    def test_available_sm_counts(self, gpu_id):
        cpp = CppBackend(gpu_id=gpu_id)
        sms = cpp.available_sm_counts()
        assert len(sms) > 0
        assert sms == sorted(sms)
        # All multiples of sm_step
        step = cpp.sm_step()
        for s in sms:
            assert s % step == 0
        cpp.close()

    def test_stream_bundles(self, gpu_id):
        cpp = CppBackend(gpu_id=gpu_id)
        sms = cpp.available_sm_counts()
        for sm in [sms[0], sms[-1]]:
            b = cpp.get_stream_bundle(sm)
            assert b.sm_count == sm
            assert isinstance(b.comp, torch.cuda.Stream)
            assert isinstance(b.recv, torch.cuda.Stream)
            assert isinstance(b.send, torch.cuda.Stream)
            assert isinstance(b.dp, torch.cuda.Stream)
        cpp.close()

    def test_no_per_step_stream_creation(self, gpu_id):
        """Verify zero-overhead: no ExternalStream created
        during step activation."""
        cpp = CppBackend(gpu_id=gpu_id)
        # After init, stream cache is populated
        cache_size = len(cpp._stream_cache)
        assert cache_size > 0

        # Activate a step
        sm, gen = cpp.activate_for_step(0)
        # Cache should not grow
        assert len(cpp._stream_cache) == cache_size

        cpp.deactivate(sm)
        cpp.close()

    def test_switch_count(self, gpu_id):
        cpp = CppBackend(gpu_id=gpu_id)
        sms = cpp.available_sm_counts()
        assert cpp.switch_count() == 0

        cpp.activate_for_step(0)
        # Without trace, always uses default
        cpp.deactivate(sms[-1])
        cpp.close()


# ── Backend selection tests ──────────────────────────────


class TestBackendSelection:
    def test_auto_selects_cpp(self, gpu_id):
        b = select_backend(backend="auto", gpu_id=gpu_id)
        assert isinstance(b, CppBackend)
        b.close()

    def test_off_backend(self, gpu_id):
        b = select_backend(backend="off", gpu_id=gpu_id)
        assert isinstance(b, OffBackend)
        b.close()

    def test_cpp_explicit(self, gpu_id):
        b = select_backend(backend="cpp", gpu_id=gpu_id)
        assert isinstance(b, CppBackend)
        b.close()


# ── Controller tests ─────────────────────────────────────


class TestGreenContextController:
    def test_enabled(self, gpu_id):
        cfg = GreenContextConfig(enabled=True)
        ctrl = GreenContextController.from_config(
            device_id=gpu_id, cfg=cfg
        )
        assert ctrl.is_supported
        ctrl.close()

    def test_disabled(self):
        cfg = GreenContextConfig(enabled=False)
        ctrl = GreenContextController.from_config(
            device_id=0, cfg=cfg
        )
        assert not ctrl.is_supported
        ctrl.close()

    def test_step_scope_returns_bundle(self, gpu_id):
        cfg = GreenContextConfig(enabled=True)
        ctrl = GreenContextController.from_config(
            device_id=gpu_id, cfg=cfg
        )
        with ctrl.step_scope(0) as bundle:
            assert isinstance(bundle, StreamBundle)
            assert bundle.sm_count > 0
            assert isinstance(
                bundle.comp, torch.cuda.Stream
            )
        ctrl.close()

    def test_step_scope_disabled(self):
        cfg = GreenContextConfig(enabled=False)
        ctrl = GreenContextController.from_config(
            device_id=0, cfg=cfg
        )
        with ctrl.step_scope(0) as bundle:
            assert bundle.sm_count == 0
        ctrl.close()


# ── Trace-driven tests ───────────────────────────────────


class TestTraceDriven:
    def test_step_sm_mapping(self, gpu_id, trace_path):
        """Verify trace: step 0→8, 1→32, 2→64, etc."""
        cfg = GreenContextConfig(
            enabled=True,
            trace_path=trace_path,
            clock_mode="step",
        )
        ctrl = GreenContextController.from_config(
            device_id=gpu_id, cfg=cfg
        )
        # Trace: 0→8, 1→32, 2→64, 3→16, 4→48, 5→64
        expected = {0: 8, 1: 32, 2: 64, 3: 16, 4: 48, 5: 64}
        actual = {}
        for step in range(6):
            with ctrl.step_scope(step) as s:
                actual[step] = s.sm_count
        assert actual == expected
        ctrl.close()

    def test_beyond_trace_holds_last(
        self, gpu_id, trace_path
    ):
        cfg = GreenContextConfig(
            enabled=True,
            trace_path=trace_path,
            clock_mode="step",
        )
        ctrl = GreenContextController.from_config(
            device_id=gpu_id, cfg=cfg
        )
        with ctrl.step_scope(100) as s:
            assert s.sm_count == 64
        ctrl.close()

    def test_switch_count_increments(
        self, gpu_id, trace_path
    ):
        cfg = GreenContextConfig(
            enabled=True,
            trace_path=trace_path,
            clock_mode="step",
        )
        ctrl = GreenContextController.from_config(
            device_id=gpu_id, cfg=cfg
        )
        for step in range(6):
            with ctrl.step_scope(step):
                pass
        assert ctrl.switch_count() >= 4
        ctrl.close()


# ── Wall-clock trace tests ──────────────────────────────


class TestWallClockTrace:
    @pytest.fixture
    def wall_trace_path(self):
        root = Path(__file__).resolve().parents[5]
        p = (
            root
            / "tests"
            / "data"
            / "greenctx"
            / "second_level_wall.trace"
        )
        if not p.exists():
            pytest.skip("Wall-clock trace not found")
        return str(p)

    def test_wall_trace_parses_time_unit(
        self, wall_trace_path
    ):
        """Verify time_unit=s is parsed from comment
        directive."""
        from morphling.runtime.green_context_backends import (
            _parse_trace_file,
        )

        entries, tu, cm = _parse_trace_file(
            wall_trace_path
        )
        assert tu == "s"
        assert len(entries) == 6
        # First entry: 0s -> 8 SMs
        assert entries[0] == (0, 8)
        # Last entry: 5s -> 64 SMs
        assert entries[-1] == (5, 64)

    def test_sm_count_at_time_seconds(
        self, wall_trace_path
    ):
        """Verify wall-clock lookup converts seconds to
        microseconds correctly."""
        from morphling.runtime.green_context_backends import (
            _parse_trace_file,
            _sm_count_at_time,
        )

        entries, tu, _cm = _parse_trace_file(
            wall_trace_path
        )
        # At t=0us -> 8 SMs (first entry at 0s)
        assert (
            _sm_count_at_time(entries, 0, tu, 64) == 8
        )
        # At t=0.5s (500000us) -> still 8 SMs
        assert (
            _sm_count_at_time(entries, 500_000, tu, 64)
            == 8
        )
        # At t=1s (1000000us) -> 32 SMs
        assert (
            _sm_count_at_time(entries, 1_000_000, tu, 64)
            == 32
        )
        # At t=2.5s (2500000us) -> 64 SMs (entry at 2s)
        assert (
            _sm_count_at_time(entries, 2_500_000, tu, 64)
            == 64
        )
        # At t=3s (3000000us) -> 16 SMs (throttle)
        assert (
            _sm_count_at_time(entries, 3_000_000, tu, 64)
            == 16
        )
        # Beyond last entry: holds 64 SMs
        assert (
            _sm_count_at_time(
                entries, 10_000_000, tu, 64
            )
            == 64
        )

    def test_wall_trace_boundary_at_zero(
        self, wall_trace_path
    ):
        """Verify t=0 maps correctly when first entry
        is at t=0."""
        from morphling.runtime.green_context_backends import (
            _parse_trace_file,
            _sm_count_at_time,
        )

        entries, tu, _cm = _parse_trace_file(
            wall_trace_path
        )
        # Exactly at boundary
        assert (
            _sm_count_at_time(entries, 0, tu, 99) == 8
        )

    def test_wall_trace_with_cpp_backend(
        self, gpu_id, wall_trace_path
    ):
        """Verify CppBackend handles wall-clock trace
        via sm_count_at_time."""
        cpp = CppBackend(gpu_id=gpu_id)
        loaded = cpp.load_trace(wall_trace_path)
        assert loaded
        # At t=0us -> 8 SMs
        assert cpp.sm_count_at_time(0) == 8
        # At t=1s = 1000000us -> 32 SMs
        assert cpp.sm_count_at_time(1_000_000) == 32
        # At t=5s = 5000000us -> 64 SMs
        assert cpp.sm_count_at_time(5_000_000) == 64
        cpp.close()


# ── Python trace parser unit tests ──────────────────────


class TestPythonTraceParser:
    def test_parse_step_trace(self, trace_path):
        from morphling.runtime.green_context_backends import (
            _parse_trace_file,
        )

        entries, tu, cm = _parse_trace_file(trace_path)
        assert tu == "us"
        assert len(entries) == 6
        assert entries[0] == (0, 8)
        assert entries[2] == (2, 64)

    def test_step_lookup_beyond_range(self):
        from morphling.runtime.green_context_backends import (
            _sm_count_at_step,
        )

        entries = [(0, 8), (5, 32)]
        # Before first: returns default
        assert _sm_count_at_step(entries, -1, 99) == 99
        # At step 0
        assert _sm_count_at_step(entries, 0, 99) == 8
        # Between entries
        assert _sm_count_at_step(entries, 3, 99) == 8
        # At step 5
        assert _sm_count_at_step(entries, 5, 99) == 32
        # Beyond last
        assert _sm_count_at_step(entries, 100, 99) == 32


# ── TorchNativeBackend tests ──────────────────────────


def _torch_native_available() -> bool:
    """Check if torch.cuda.green_contexts is importable
    and SUPPORTED is True."""
    try:
        from torch.cuda.green_contexts import SUPPORTED
        return bool(SUPPORTED)
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(
    not _torch_native_available(),
    reason="torch.cuda.green_contexts not available"
)
class TestTorchNativeBackend:
    """Tests for TorchNativeBackend (PyTorch >= 2.10)."""

    @pytest.fixture(scope="class")
    def native_gpu_id(self):
        """GPU fixture independent of CppBackend."""
        gid = _get_idle_gpu()
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        b = TorchNativeBackend(gpu_id=gid)
        if not b.is_supported():
            b.close()
            pytest.skip(
                "TorchNativeBackend not supported on "
                f"GPU {gid}"
            )
        b.close()
        return gid

    def test_supported(self, native_gpu_id):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(gpu_id=native_gpu_id)
        assert backend.is_supported()
        assert backend.unsupported_reason() == ""
        backend.close()

    def test_sm_step(self, native_gpu_id):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(gpu_id=native_gpu_id)
        step = backend.sm_step()
        assert step in (2, 8)  # CC 8.x=2, CC 9.0+=8
        backend.close()

    def test_available_sm_counts(self, native_gpu_id):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(gpu_id=native_gpu_id)
        counts = backend.available_sm_counts()
        assert len(counts) > 0
        step = backend.sm_step()
        for c in counts:
            assert c % step == 0
        backend.close()

    def test_stream_bundles(self, native_gpu_id):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(gpu_id=native_gpu_id)
        counts = backend.available_sm_counts()
        sm = counts[0]
        bundle = backend.get_stream_bundle(sm)
        assert isinstance(bundle, StreamBundle)
        assert bundle.sm_count == sm
        assert isinstance(bundle.comp, torch.cuda.Stream)
        assert isinstance(bundle.recv, torch.cuda.Stream)
        assert isinstance(bundle.send, torch.cuda.Stream)
        assert isinstance(bundle.dp, torch.cuda.Stream)
        backend.close()

    def test_activate_deactivate(self, native_gpu_id):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(
            gpu_id=native_gpu_id
        )
        counts = backend.available_sm_counts()
        sm = counts[0]
        # Activate directly (no trace loaded)
        result_sm, gen = backend.activate_for_step(0)
        # Without trace, should use partition_sm
        assert result_sm == backend.partition_sm_count()
        backend.deactivate(result_sm)
        backend.close()

    def test_load_trace_and_step(
        self, native_gpu_id, trace_path
    ):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(
            gpu_id=native_gpu_id
        )
        loaded = backend.load_trace(trace_path)
        assert loaded
        # Step 0 -> 8 SMs (from trace)
        sm0 = backend.sm_count_at_step(0)
        assert sm0 == 8
        # Step 1 -> 32 SMs
        sm1 = backend.sm_count_at_step(1)
        assert sm1 == 32
        backend.close()

    def test_switch_count(
        self, native_gpu_id, trace_path
    ):
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        backend = TorchNativeBackend(
            gpu_id=native_gpu_id
        )
        loaded = backend.load_trace(trace_path)
        assert loaded

        # Activate step 0 (8 SMs)
        sm0, _ = backend.activate_for_step(0)
        backend.deactivate(sm0)

        # Activate step 1 (32 SMs) — different SM
        sm1, _ = backend.activate_for_step(1)
        assert sm1 != sm0  # 32 != 8
        backend.deactivate(sm1)

        assert backend.switch_count() >= 1
        backend.close()

    def test_select_backend_torch_native(
        self, native_gpu_id
    ):
        """select_backend('torch_native') picks this."""
        b = select_backend(
            backend="torch_native",
            gpu_id=native_gpu_id,
        )
        from morphling.runtime.green_context_backends import (
            TorchNativeBackend,
        )

        assert isinstance(b, TorchNativeBackend)
        assert b.is_supported()
        b.close()
