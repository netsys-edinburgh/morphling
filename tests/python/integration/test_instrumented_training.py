from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_greenctx_training.py"
ANALYZE_SCRIPT = REPO_ROOT / "baselines" / "examples" / "analyze_violations.py"


def _run_python(cmd: list[str], *, timeout_s: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _write_autograd_fixture(path: Path) -> None:
    payload = [
        {
            "step": 0,
            "phase": "forward",
            "start_us": 0.0,
            "end_us": 300.0,
            "duration_us": 300.0,
            "sm_count": 8,
            "greenctx_enabled": True,
            "m": 64,
            "n": 64,
            "k": 64,
        },
        {
            "step": 1,
            "phase": "forward",
            "start_us": 1100.0,
            "end_us": 1150.0,
            "duration_us": 50.0,
            "sm_count": 16,
            "greenctx_enabled": True,
            "m": 64,
            "n": 64,
            "k": 64,
        },
    ]
    _ = path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trace_fixture(path: Path) -> None:
    rows = [
        ["time_slot_sched_ns", "sm_count", "time_decode_start_actual_ns"],
        [0, 32, 100],
        [200_000, 24, 200_100],
        [1_000_000, 24, 1_000_100],
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _assert_metrics_columns(metrics_csv: Path) -> None:
    assert metrics_csv.exists(), f"missing metrics CSV: {metrics_csv}"
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "step",
        "wall_time_ms",
        "tokens_per_sec",
        "sm_count",
        "swap_count",
        "swap_overhead_us",
        "loss",
        "gpu_mem_mb",
    }
    assert required.issubset(cols)
    assert rows, "metrics CSV is empty"


def _assert_swap_stats_shape(greenctx_csv: Path) -> None:
    assert greenctx_csv.exists(), f"missing greenctx metrics CSV: {greenctx_csv}"
    with greenctx_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, "greenctx metrics CSV is empty"

    saw_swap_field = False
    for row in rows:
        if "swap_count" in row and "swap_overhead_us" in row:
            saw_swap_field = True
            assert float(row["swap_count"]) >= 0.0
            assert float(row["swap_overhead_us"]) >= 0.0
    assert saw_swap_field, "swap stat fields not present"


def test_instrumented_training_pipeline_smoke(tmp_path: Path) -> None:
    if not EVAL_SCRIPT.exists():
        pytest.skip(f"missing script: {EVAL_SCRIPT}")
    if not ANALYZE_SCRIPT.exists():
        pytest.skip(f"missing script: {ANALYZE_SCRIPT}")

    out_dir = tmp_path / "instrumented"
    result = _run_python(
        [
            str(EVAL_SCRIPT),
            "--num-steps",
            "3",
            "--output-dir",
            str(out_dir),
        ],
        timeout_s=180,
    )

    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        skip_markers = (
            "cuda is required",
            "no cuda",
            "morphling._msg",
            "librabbitmq.so",
            "out of memory",
        )
        if any(marker in stderr for marker in skip_markers):
            pytest.skip(f"instrumented run unavailable in this env: {result.stderr.strip()}")
        pytest.fail(
            "eval_greenctx_training.py failed\n"
            + f"stdout:\n{result.stdout}\n"
            + f"stderr:\n{result.stderr}"
        )

    metrics_csv = out_dir / "metrics.csv"
    greenctx_csv = out_dir / "eval_metrics_greenctx.csv"
    _assert_metrics_columns(metrics_csv)
    _assert_swap_stats_shape(greenctx_csv)

    gemm_logs = sorted(out_dir.glob("gemm_log_step_*.json"))
    assert gemm_logs, "expected per-step GEMM log dumps"

    autograd_log = tmp_path / "autograd_log.json"
    trace_csv = tmp_path / "trace.csv"
    analysis_json = tmp_path / "violations.json"
    _write_autograd_fixture(autograd_log)
    _write_trace_fixture(trace_csv)

    analyze = _run_python(
        [
            str(ANALYZE_SCRIPT),
            "--mode",
            "autograd",
            "--autograd-log",
            str(autograd_log),
            "--trace",
            str(trace_csv),
            "--output-json",
            str(analysis_json),
        ],
        timeout_s=60,
    )
    assert analyze.returncode == 0, (
        "analyze_violations.py failed\n"
        f"stdout:\n{analyze.stdout}\n"
        f"stderr:\n{analyze.stderr}"
    )
    assert analysis_json.exists()

    data = cast(dict[str, object], json.loads(analysis_json.read_text(encoding="utf-8")))
    assert "total_violations" in data
    assert "total_violation_time_ms" in data
    assert "violations_per_step" in data
    assert "avg_violation_duration_us" in data
    assert "per_slot_summary" in data
