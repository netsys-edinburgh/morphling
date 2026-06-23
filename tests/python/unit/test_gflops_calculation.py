#!/usr/bin/env python3
"""Unit tests for GFLOPS benchmark script."""

import subprocess
import sys


class TestBenchGflopsScript:
    def test_script_help(self):
        result = subprocess.run(
            [sys.executable, "scripts/bench_gflops_per_sm.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--gemm-shapes" in result.stdout
        assert "--sm-counts" in result.stdout

    def test_flops_formula(self):
        m, n, k = 128, 256, 64
        flops = 2.0 * m * n * k
        time_ms = 0.5
        gflops = flops / (time_ms * 1e-3) / 1e9
        assert abs(gflops - 8.388608) < 0.01
