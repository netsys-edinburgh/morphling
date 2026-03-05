#!/usr/bin/env python3
"""Unit tests for swap timing instrumentation source-level checks.

These tests avoid importing morphling package init, because it requires
compiled extension modules that may be unavailable in CI.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKENDS = ROOT / "morphling" / "runtime" / "green_context_backends.py"
CONTROLLER = ROOT / "morphling" / "runtime" / "green_context.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSwapTimingSource:
    def test_cpp_backend_has_swap_tracking_fields(self):
        src = _read(BACKENDS)
        assert "self._swap_log" in src
        assert "self._swap_count" in src
        assert "self._total_python_overhead_ns" in src
        assert "self._prev_sm_count" in src

    def test_cpp_backend_uses_perf_counter(self):
        src = _read(BACKENDS)
        assert "time.perf_counter_ns()" in src
        assert "activate_for_time" in src
        assert "deactivate" in src

    def test_controller_exposes_swap_api(self):
        src = _read(CONTROLLER)
        assert "def get_swap_stats" in src
        assert "def get_swap_log" in src
        assert "def reset_swap_stats" in src
        assert "python_layer" in src
