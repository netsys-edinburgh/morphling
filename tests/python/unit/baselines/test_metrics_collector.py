from __future__ import annotations

import json
from pathlib import Path

from baselines.utils.metrics import MetricsCollector


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_metrics_collector_emits_cumulative_fields(
    tmp_path: Path,
) -> None:
    mc = MetricsCollector(
        rank=0, out_dir=str(tmp_path), enabled=True
    )
    mc.iter_start(0)
    mc.set_cumulative_timings(
        pp_send_cumulative_ms=12.5,
        pp_recv_cumulative_ms=6.25,
        forward_compute_ms=31.0,
        backward_compute_ms=17.5,
    )
    mc.iter_end()
    mc.close()

    rows = _read_jsonl(tmp_path / "rank_0.jsonl")
    assert len(rows) == 1
    step = rows[0]
    assert step["type"] == "step"
    assert step["pp_send_cumulative_ms"] == 12.5
    assert step["pp_recv_cumulative_ms"] == 6.25
    assert step["forward_compute_ms"] == 31.0
    assert step["backward_compute_ms"] == 17.5


def test_metrics_collector_resets_cumulative_fields_each_iter(
    tmp_path: Path,
) -> None:
    mc = MetricsCollector(
        rank=1, out_dir=str(tmp_path), enabled=True
    )
    mc.iter_start(0)
    mc.set_cumulative_timings(
        pp_send_cumulative_ms=9.0,
        pp_recv_cumulative_ms=8.0,
        forward_compute_ms=7.0,
        backward_compute_ms=6.0,
    )
    mc.iter_end()

    mc.iter_start(1)
    mc.iter_end()
    mc.close()

    rows = _read_jsonl(tmp_path / "rank_1.jsonl")
    assert len(rows) == 2
    first = rows[0]
    second = rows[1]
    assert first["pp_send_cumulative_ms"] == 9.0
    assert first["pp_recv_cumulative_ms"] == 8.0
    assert first["forward_compute_ms"] == 7.0
    assert first["backward_compute_ms"] == 6.0
    assert second["pp_send_cumulative_ms"] == 0.0
    assert second["pp_recv_cumulative_ms"] == 0.0
    assert second["forward_compute_ms"] == 0.0
    assert second["backward_compute_ms"] == 0.0


def test_metrics_collector_emits_micro_ops(
    tmp_path: Path,
) -> None:
    mc = MetricsCollector(
        rank=2, out_dir=str(tmp_path), enabled=True
    )
    mc.iter_start(3)
    mc.log_micro_ops(
        3,
        [
            {
                "microbatch": 0,
                "op_seq": 0,
                "op_kind": "fwd_compute",
                "stream": "comp",
                "start_offset_ms": 0.1,
                "end_offset_ms": 1.2,
                "duration_ms": 1.1,
                "nbytes": 0,
                "peer_rank": None,
            },
            {
                "microbatch": 0,
                "op_seq": 1,
                "op_kind": "fwd_pp_send",
                "stream": "send",
                "start_offset_ms": 1.3,
                "end_offset_ms": 2.0,
                "duration_ms": 0.7,
                "nbytes": 1024,
                "peer_rank": 5,
            },
        ],
    )
    mc.iter_end()
    mc.close()

    rows = _read_jsonl(tmp_path / "rank_2.jsonl")
    micro_rows = [
        r for r in rows if r.get("type") == "micro_op"
    ]
    assert len(micro_rows) == 2
    assert micro_rows[0]["iter"] == 3
    assert micro_rows[0]["microbatch"] == 0
    assert micro_rows[0]["op_kind"] == "fwd_compute"
    assert micro_rows[1]["nbytes"] == 1024
    assert micro_rows[1]["peer_rank"] == 5
