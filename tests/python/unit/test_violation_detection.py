#!/usr/bin/env python3
"""Unit tests for violation detection (dual-mode)."""
import json
import subprocess
import sys
import pytest

SCRIPT = "baselines/examples/analyze_violations.py"


class TestViolationAnalysisCLI:
    def test_help_includes_mode(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert "--mode" in result.stdout
        assert "autograd" in result.stdout
        assert "ldpreload" in result.stdout

    def test_help_includes_output_json(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True, text=True,
        )
        assert "--output-json" in result.stdout


class TestAutogradMode:
    def test_autograd_with_synthetic_data(self, tmp_path):
        gemm_log = [
            {"gemm_idx": 0, "phase": "forward",
             "start_us": 100.0, "end_us": 200.0,
             "duration_us": 100.0, "sm_count": 8,
             "greenctx_enabled": True, "m": 64, "n": 128, "k": 64},
        ]
        log_path = tmp_path / "gemm_log.json"
        log_path.write_text(json.dumps(gemm_log))
        trace_path = tmp_path / "trace.csv"
        trace_path.write_text(
            "time_slot_sched_ns,sm_count,time_decode_start_actual_ns\n"
            "100000,8,150000\n"
            "300000,16,350000\n"
        )
        output_json = tmp_path / "violations.json"
        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--mode", "autograd",
             "--autograd-log", str(log_path),
             "--trace", str(trace_path),
             "--output-json", str(output_json)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_json.exists()
        data = json.loads(output_json.read_text())
        assert "total_slots" in data
        assert "total_gemms" in data
