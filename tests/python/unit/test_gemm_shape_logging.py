#!/usr/bin/env python3
"""Unit tests for GEMM shape logging in autograd hook.

Loads autograd.py directly from file with minimal module stubs to avoid
importing morphling package __init__ (which needs compiled extensions).
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUTOGRAD_PATH = ROOT / "morphling" / "hooks" / "autograd.py"


def _load_autograd_module():
    # Minimal stub for `from morphling.backend import BaseBackend`
    backend_mod = types.ModuleType("morphling.backend")

    class BaseBackend:  # noqa: D401 - simple stub
        """Stub backend base class."""

    backend_mod.BaseBackend = BaseBackend

    morphling_pkg = types.ModuleType("morphling")
    morphling_pkg.__path__ = [str(ROOT / "morphling")]

    sys.modules.setdefault("morphling", morphling_pkg)
    sys.modules["morphling.backend"] = backend_mod

    spec = importlib.util.spec_from_file_location(
        "morphling.hooks.autograd_test", str(AUTOGRAD_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestGemmLogFormat:
    def test_log_gemm_includes_dimensions(self):
        autograd = _load_autograd_module()

        autograd._gemm_log.clear()
        autograd._gemm_idx = 0
        autograd._log_gemm(
            gemm_idx=0,
            phase="forward",
            start_us=100.0,
            end_us=200.0,
            sm_count=8,
            m=64,
            n=128,
            k=256,
        )

        log = autograd.get_gemm_log()
        assert len(log) == 1
        e = log[0]
        assert e["m"] == 64
        assert e["n"] == 128
        assert e["k"] == 256
        assert e["phase"] == "forward"
        assert e["duration_us"] == 100.0

    def test_log_gemm_default_dimensions(self):
        autograd = _load_autograd_module()

        autograd._gemm_log.clear()
        autograd._gemm_idx = 0
        autograd._log_gemm(
            gemm_idx=0,
            phase="backward",
            start_us=0.0,
            end_us=50.0,
            sm_count=16,
        )

        log = autograd.get_gemm_log()
        assert log[0]["m"] == 0
        assert log[0]["n"] == 0
        assert log[0]["k"] == 0

    def test_get_gemm_log_reset(self):
        autograd = _load_autograd_module()

        autograd._gemm_log.clear()
        autograd._log_gemm(
            gemm_idx=0,
            phase="fwd",
            start_us=0.0,
            end_us=1.0,
            sm_count=8,
            m=1,
            n=2,
            k=3,
        )

        log = autograd.get_gemm_log(reset=True)
        assert len(log) == 1
        assert len(autograd._gemm_log) == 0
