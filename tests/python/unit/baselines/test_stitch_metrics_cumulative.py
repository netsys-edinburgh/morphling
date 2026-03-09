from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from baselines.scripts.stitch_metrics import (
    _ensure_schema,
    stitch_experiment,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_stitch_metrics_populates_new_cumulative_columns(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)

    data_dir = tmp_path / "exp_new"
    data_dir.mkdir(parents=True)

    r0 = [
        {
            "type": "step",
            "rank": 0,
            "iter": 0,
            "start_ts": "2026-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:01+00:00",
            "forward_start_ts": "2026-01-01T00:00:00+00:00",
            "forward_end_ts": "2026-01-01T00:00:00.500000+00:00",
            "backward_start_ts": "2026-01-01T00:00:00.500000+00:00",
            "backward_end_ts": "2026-01-01T00:00:00.900000+00:00",
            "pp_send_start_ts": "2026-01-01T00:00:00.100000+00:00",
            "pp_send_end_ts": "2026-01-01T00:00:00.400000+00:00",
            "pp_recv_start_ts": "2026-01-01T00:00:00.150000+00:00",
            "pp_recv_end_ts": "2026-01-01T00:00:00.450000+00:00",
            "pp_send_bytes": 1024,
            "dp_allreduce_start_ts": "2026-01-01T00:00:00.900000+00:00",
            "dp_allreduce_end_ts": "2026-01-01T00:00:00.950000+00:00",
            "dp_allreduce_bytes": 2048,
            "pp_send_cumulative_ms": 5.0,
            "pp_recv_cumulative_ms": 4.0,
            "forward_compute_ms": 11.0,
            "backward_compute_ms": 22.0,
        },
        {
            "type": "global_step_raw",
            "iter": 0,
            "loss": 1.0,
            "lr": 1e-4,
            "iter_start_ts": "2026-01-01T00:00:00+00:00",
            "iter_end_ts": "2026-01-01T00:00:01+00:00",
        },
    ]
    r1 = [
        {
            "type": "step",
            "rank": 1,
            "iter": 0,
            "start_ts": "2026-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:01.200000+00:00",
            "forward_start_ts": "2026-01-01T00:00:00+00:00",
            "forward_end_ts": "2026-01-01T00:00:00.600000+00:00",
            "backward_start_ts": "2026-01-01T00:00:00.600000+00:00",
            "backward_end_ts": "2026-01-01T00:00:01+00:00",
            "pp_send_start_ts": "2026-01-01T00:00:00.050000+00:00",
            "pp_send_end_ts": "2026-01-01T00:00:00.250000+00:00",
            "pp_recv_start_ts": "2026-01-01T00:00:00.300000+00:00",
            "pp_recv_end_ts": "2026-01-01T00:00:00.500000+00:00",
            "pp_send_bytes": 512,
            "dp_allreduce_start_ts": "2026-01-01T00:00:01+00:00",
            "dp_allreduce_end_ts": "2026-01-01T00:00:01.050000+00:00",
            "dp_allreduce_bytes": 1024,
            "pp_send_cumulative_ms": 3.0,
            "pp_recv_cumulative_ms": 2.0,
            "forward_compute_ms": 7.0,
            "backward_compute_ms": 9.0,
        }
    ]
    _write_jsonl(data_dir / "rank_0.jsonl", r0)
    _write_jsonl(data_dir / "rank_1.jsonl", r1)

    stitch_experiment(
        experiment_id="exp-new",
        data_dir=data_dir,
        conn=conn,
    )

    node = conn.execute(
        """SELECT
               pp_send_cumulative_ms,
               pp_recv_cumulative_ms,
               forward_compute_ms,
               backward_compute_ms
           FROM step_metrics
           WHERE experiment_id='exp-new' AND rank=0 AND iter=0"""
    ).fetchone()
    assert node == (5.0, 4.0, 11.0, 22.0)

    global_row = conn.execute(
        """SELECT
               total_pp_send_cumulative_ms,
               total_pp_recv_cumulative_ms,
               max_pp_send_cumulative_ms,
               max_pp_recv_cumulative_ms,
               total_forward_compute_ms,
               total_backward_compute_ms
           FROM global_step_metrics
           WHERE experiment_id='exp-new' AND iter=0"""
    ).fetchone()
    assert global_row == (8.0, 6.0, 5.0, 4.0, 18.0, 31.0)


def test_stitch_metrics_legacy_records_leave_new_columns_null(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)

    data_dir = tmp_path / "exp_legacy"
    data_dir.mkdir(parents=True)
    _write_jsonl(
        data_dir / "rank_0.jsonl",
        [
            {
                "type": "step",
                "rank": 0,
                "iter": 0,
                "start_ts": "2026-01-01T00:00:00+00:00",
                "end_ts": "2026-01-01T00:00:01+00:00",
            }
        ],
    )

    stitch_experiment(
        experiment_id="exp-legacy",
        data_dir=data_dir,
        conn=conn,
    )

    row = conn.execute(
        """SELECT
               pp_send_cumulative_ms,
               pp_recv_cumulative_ms,
               forward_compute_ms,
               backward_compute_ms
           FROM step_metrics
           WHERE experiment_id='exp-legacy' AND rank=0 AND iter=0"""
    ).fetchone()
    assert row == (None, None, None, None)


def test_stitch_metrics_preserves_explicit_zero_cumulative_values(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)

    data_dir = tmp_path / "exp_zero"
    data_dir.mkdir(parents=True)
    _write_jsonl(
        data_dir / "rank_0.jsonl",
        [
            {
                "type": "step",
                "rank": 0,
                "iter": 0,
                "start_ts": "2026-01-01T00:00:00+00:00",
                "end_ts": "2026-01-01T00:00:01+00:00",
                "pp_send_cumulative_ms": 0.0,
                "pp_recv_cumulative_ms": 0.0,
                "forward_compute_ms": 0.0,
                "backward_compute_ms": 0.0,
            }
        ],
    )

    stitch_experiment(
        experiment_id="exp-zero",
        data_dir=data_dir,
        conn=conn,
    )

    row = conn.execute(
        """SELECT
               total_pp_send_cumulative_ms,
               total_pp_recv_cumulative_ms,
               max_pp_send_cumulative_ms,
               max_pp_recv_cumulative_ms,
               total_forward_compute_ms,
               total_backward_compute_ms
           FROM global_step_metrics
           WHERE experiment_id='exp-zero' AND iter=0"""
    ).fetchone()
    assert row == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
