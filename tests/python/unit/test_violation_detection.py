#!/usr/bin/env python3
"""Unit tests for violation detection (dual-mode)."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

SCRIPT = "scripts/analyze_violations.py"

REQUIRED_SCHEMA_KEYS = {
    "total_violations",
    "total_violation_time_ms",
    "violations_per_step",
    "avg_violation_duration_us",
    "per_slot_summary",
}

LEGACY_COMPAT_KEYS = {
    "total_slots",
    "slots_with_violations",
    "violation_pct",
    "total_gemms",
    "violating_gemms",
    "total_violation_time_ns",
    "per_step_violations",
}


class TestViolationAnalysisCLI:
    def test_help_includes_mode(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert "--mode" in result.stdout
        assert "autograd" in result.stdout
        assert "ldpreload" in result.stdout

    def test_help_includes_output_json(self):
        result = subprocess.run(
            [sys.executable, SCRIPT, "--help"],
            capture_output=True,
            text=True,
        )
        assert "--output-json" in result.stdout


def _assert_required_and_legacy_schema(data: dict[str, object]) -> None:
    assert REQUIRED_SCHEMA_KEYS.issubset(data.keys())
    assert LEGACY_COMPAT_KEYS.issubset(data.keys())

    assert isinstance(data["total_violations"], int)
    assert isinstance(data["total_violation_time_ms"], (int, float))
    assert isinstance(data["avg_violation_duration_us"], (int, float))

    assert isinstance(data["violations_per_step"], list)
    assert isinstance(data["per_slot_summary"], list)


def _load_json_dict(path: Path) -> dict[str, object]:
    parsed = cast(object, json.loads(path.read_text()))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


class TestAutogradMode:
    def test_autograd_with_synthetic_data(self, tmp_path: Path) -> None:
        gemm_log = [
            {
                "gemm_idx": 0,
                "phase": "forward",
                "start_us": 100.0,
                "end_us": 200.0,
                "duration_us": 100.0,
                "sm_count": 8,
                "greenctx_enabled": True,
                "m": 64,
                "n": 128,
                "k": 64,
            },
        ]
        log_path = tmp_path / "gemm_log.json"
        _ = log_path.write_text(json.dumps(gemm_log))
        trace_path = tmp_path / "trace.csv"
        _ = trace_path.write_text(
            "time_slot_sched_ns,sm_count,time_decode_start_actual_ns\n"
            + "100000,8,150000\n"
            + "300000,16,350000\n"
        )
        output_json = tmp_path / "violations.json"
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--mode",
                "autograd",
                "--autograd-log",
                str(log_path),
                "--trace",
                str(trace_path),
                "--output-json",
                str(output_json),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_json.exists()
        data = _load_json_dict(output_json)
        _assert_required_and_legacy_schema(data)
        assert data["total_gemms"] == 1


class TestLdpreloadMode:
    def test_ldpreload_with_synthetic_csv_logs(
        self,
        tmp_path: Path,
    ) -> None:
        gemm_log = tmp_path / "gemm_log.csv"
        _ = gemm_log.write_text(
            "function_name,start_ns,end_ns,duration_ns,m,n,k\n"
            + "cublasGemmEx,100000,260000,160000,64,128,64\n"
        )

        step_log = tmp_path / "step_log.csv"
        _ = step_log.write_text(
            "step,start_ns,end_ns,sm_count\n"
            + "0,0,200000,32\n"
            + "1,200000,350000,32\n"
            + "2,350000,500000,16\n"
        )

        output_json = tmp_path / "violations.json"
        result = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--mode",
                "ldpreload",
                "--gemm-log",
                str(gemm_log),
                "--step-log",
                str(step_log),
                "--max-sm-count",
                "48",
                "--output-json",
                str(output_json),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_json.exists()

        data = _load_json_dict(output_json)
        _assert_required_and_legacy_schema(data)

        total_violations = cast(int, data["total_violations"])
        total_violation_time_ms = cast(
            float,
            data["total_violation_time_ms"],
        )
        violations_per_step = cast(
            list[object],
            data["violations_per_step"],
        )
        per_slot_summary = cast(
            list[object],
            data["per_slot_summary"],
        )

        assert total_violations >= 1
        assert total_violation_time_ms >= 0
        assert len(violations_per_step) >= 1
        assert len(per_slot_summary) >= 1
