from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from baselines.scripts.stitch_metrics import (
    _ensure_schema,
    _interval_union_ms,
    stitch_experiment,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_interval_union_ms_merges_overlaps() -> None:
    assert _interval_union_ms([]) == 0.0
    assert _interval_union_ms([(0.0, 1.0)]) == 1.0
    assert _interval_union_ms([(0.0, 2.0), (1.0, 3.0)]) == 3.0
    assert _interval_union_ms([(0.0, 1.0), (2.0, 3.0)]) == 2.0
    assert _interval_union_ms([(0.0, 5.0), (1.0, 2.0)]) == 5.0


def test_stitch_microbatch_tables_populate(
    tmp_path: Path,
) -> None:
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)

    data_dir = tmp_path / "exp_micro"
    data_dir.mkdir(parents=True)

    rank0_rows = [
        {
            "type": "step",
            "rank": 0,
            "iter": 0,
            "start_ts": "2026-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:01+00:00",
        },
        {
            "type": "global_step_raw",
            "iter": 0,
            "loss": 1.0,
            "lr": 1e-4,
            "iter_start_ts": "2026-01-01T00:00:00+00:00",
            "iter_end_ts": "2026-01-01T00:00:01+00:00",
        },
        {
            "type": "micro_op",
            "rank": 0,
            "iter": 0,
            "microbatch": 0,
            "op_seq": 0,
            "op_kind": "fwd_compute",
            "stream": "comp",
            "start_offset_ms": 0.0,
            "end_offset_ms": 4.0,
            "duration_ms": 4.0,
            "nbytes": 0,
            "peer_rank": None,
        },
        {
            "type": "micro_op",
            "rank": 0,
            "iter": 0,
            "microbatch": 0,
            "op_seq": 1,
            "op_kind": "fwd_pp_send",
            "stream": "send",
            "start_offset_ms": 3.0,
            "end_offset_ms": 5.0,
            "duration_ms": 2.0,
            "nbytes": 128,
            "peer_rank": 1,
        },
    ]
    rank1_rows = [
        {
            "type": "step",
            "rank": 1,
            "iter": 0,
            "start_ts": "2026-01-01T00:00:00+00:00",
            "end_ts": "2026-01-01T00:00:01+00:00",
        },
        {
            "type": "micro_op",
            "rank": 1,
            "iter": 0,
            "microbatch": 0,
            "op_seq": 0,
            "op_kind": "fwd_compute",
            "stream": "comp",
            "start_offset_ms": 0.0,
            "end_offset_ms": 2.0,
            "duration_ms": 2.0,
            "nbytes": 0,
            "peer_rank": None,
        },
        {
            "type": "micro_op",
            "rank": 1,
            "iter": 0,
            "microbatch": 0,
            "op_seq": 1,
            "op_kind": "fwd_pp_recv",
            "stream": "recv",
            "start_offset_ms": 1.0,
            "end_offset_ms": 3.0,
            "duration_ms": 2.0,
            "nbytes": 128,
            "peer_rank": 0,
        },
    ]

    _write_jsonl(data_dir / "rank_0.jsonl", rank0_rows)
    _write_jsonl(data_dir / "rank_1.jsonl", rank1_rows)

    stitch_experiment(
        experiment_id="exp-micro",
        data_dir=data_dir,
        conn=conn,
    )

    raw_count = conn.execute(
        "SELECT COUNT(*) FROM microbatch_op_metrics "
        "WHERE experiment_id='exp-micro'"
    ).fetchone()[0]
    assert raw_count == 4

    rank0 = conn.execute(
        "SELECT compute_nonoverlap_ms, comm_nonoverlap_ms, "
        "total_union_ms, compute_sum_ms, comm_sum_ms "
        "FROM microbatch_metrics "
        "WHERE experiment_id='exp-micro' AND rank=0 AND iter=0 AND microbatch=0"
    ).fetchone()
    assert rank0 == (4.0, 2.0, 5.0, 4.0, 2.0)

    global_row = conn.execute(
        "SELECT total_compute_nonoverlap_ms, total_comm_nonoverlap_ms, "
        "total_union_ms, max_total_union_ms "
        "FROM global_microbatch_metrics "
        "WHERE experiment_id='exp-micro' AND iter=0 AND microbatch=0"
    ).fetchone()
    assert global_row == (6.0, 4.0, 8.0, 5.0)
