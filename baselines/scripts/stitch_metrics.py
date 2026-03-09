#!/usr/bin/env python3
"""Post-training JSONL stitcher.

Reads per-rank JSONL files, computes derived metrics
(durations, aggregations, straggler ratios), and populates
the SQLite experiment database.

Usage
-----
    # Stitch a single experiment
    python3 baselines/scripts/stitch_metrics.py \
        --experiment-id abc123 \
        --data-dir experiment_data/abc123 \
        --db experiments.db

    # Stitch all experiments in output dir
    python3 baselines/scripts/stitch_metrics.py \
        --all \
        --output-dir experiment_data \
        --db experiments.db
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Schema ───────────────────────────────────────────────

# Expected column counts for *metrics tables only*.
# We intentionally never auto-drop metadata tables
# (`experiments`, `rank_config`) from stitcher-side
# migration, because those are authored by
# run_experiments.py and should be preserved.
_EXPECTED_COLS: dict[str, int] = {
    "load_metrics":              9,
    "step_metrics":             38,
    "microbatch_op_metrics":    12,
    "microbatch_metrics":       10,
    "global_load_metrics":      10,
    "global_step_metrics":      27,
    "global_microbatch_metrics": 15,
}


def _migrate_tables(conn: sqlite3.Connection) -> None:
    """Drop stale metrics tables with schema drift.

    Metadata tables (`experiments`, `rank_config`) are
    never dropped here.
    """
    for table, expected in _EXPECTED_COLS.items():
        rows = conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
        if rows and len(rows) != expected:
            print(
                f"  [migrate] {table}: "
                f"{len(rows)} cols → dropping "
                f"(expected {expected})",
                flush=True,
            )
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist.

    Mirrors run_experiments.py init_db() exactly so that
    stitch_metrics.py works standalone against any DB.
    Call _migrate_tables() first to drop stale schemas.
    """
    _migrate_tables(conn)
    conn.executescript("""
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    scheduler           TEXT NOT NULL
                        CHECK (scheduler IN ('dtfm', 'confident', 'asteroid')),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    failure_reason      TEXT,
    git_commit          TEXT,
    docker_image        TEXT,
    model_name          TEXT NOT NULL,
    model_type          TEXT NOT NULL,
    num_layers          INTEGER NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    num_heads           INTEGER NOT NULL,
    d_ff                INTEGER NOT NULL,
    vocab_size          INTEGER NOT NULL,
    max_seq_len         INTEGER NOT NULL,
    global_batch_size   INTEGER NOT NULL,
    micro_batch_size    INTEGER NOT NULL,
    num_microbatches    INTEGER NOT NULL,
    lr                  REAL NOT NULL,
    min_lr              REAL NOT NULL,
    weight_decay        REAL NOT NULL,
    max_iters           INTEGER NOT NULL,
    warmup_iters        INTEGER NOT NULL,
    grad_clip           REAL NOT NULL,
    dataset             TEXT NOT NULL,
    synthetic_samples   INTEGER NOT NULL DEFAULT 2000,
    schedule_type       TEXT NOT NULL,
    pp_size             INTEGER NOT NULL,
    dp_size             INTEGER NOT NULL,
    world_size          INTEGER NOT NULL,
    comm_backend        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rank_config (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    rank                INTEGER NOT NULL,
    hostname            TEXT NOT NULL,
    ip                  TEXT NOT NULL,
    gpu_id              INTEGER NOT NULL DEFAULT 0,
    pp_stage            INTEGER NOT NULL,
    dp_group            INTEGER NOT NULL,
    mps_pct             INTEGER NOT NULL
                        CHECK (mps_pct IN (20, 40, 60, 80)),
    memory_limit_mb     INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, rank)
);

CREATE TABLE IF NOT EXISTS load_metrics (
    experiment_id           TEXT,
    rank                    INTEGER,
    model_load_start_ts     TIMESTAMP,
    model_load_end_ts       TIMESTAMP,
    duration_ms             REAL,
    nccl_init_start_ts      TIMESTAMP,
    nccl_init_end_ts        TIMESTAMP,
    nccl_duration_ms        REAL,
    memory_after_load_mb    REAL,
    PRIMARY KEY (experiment_id, rank),
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS step_metrics (
    experiment_id           TEXT,
    rank                    INTEGER,
    iter                    INTEGER,
    start_ts                TIMESTAMP,
    end_ts                  TIMESTAMP,
    duration_ms             REAL,
    forward_start_ts        TIMESTAMP,
    forward_end_ts          TIMESTAMP,
    forward_ms              REAL,
    backward_start_ts       TIMESTAMP,
    backward_end_ts         TIMESTAMP,
    backward_ms             REAL,
    optimizer_start_ts      TIMESTAMP,
    optimizer_end_ts        TIMESTAMP,
    optimizer_step_ms       REAL,
    pp_send_start_ts        TIMESTAMP,
    pp_send_end_ts          TIMESTAMP,
    pp_send_ms              REAL,
    pp_recv_start_ts        TIMESTAMP,
    pp_recv_end_ts          TIMESTAMP,
    pp_recv_ms              REAL,
    pp_send_cumulative_ms   REAL,
    pp_recv_cumulative_ms   REAL,
    forward_compute_ms      REAL,
    backward_compute_ms     REAL,
    pp_send_count           INTEGER,
    pp_recv_count           INTEGER,
    pp_send_bytes           BIGINT,
    pp_recv_bytes           BIGINT,
    dp_allreduce_start_ts   TIMESTAMP,
    dp_allreduce_end_ts     TIMESTAMP,
    dp_allreduce_ms         REAL,
    dp_allreduce_bytes      BIGINT,
    loss                    REAL,
    lr                      REAL,
    memory_allocated_mb     REAL,
    memory_reserved_mb      REAL,
    memory_peak_mb          REAL,
    PRIMARY KEY (experiment_id, rank, iter),
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS microbatch_op_metrics (
    experiment_id           TEXT,
    rank                    INTEGER,
    iter                    INTEGER,
    microbatch              INTEGER,
    op_seq                  INTEGER,
    op_kind                 TEXT,
    stream                  TEXT,
    start_offset_ms         REAL,
    end_offset_ms           REAL,
    duration_ms             REAL,
    nbytes                  BIGINT,
    peer_rank               INTEGER,
    PRIMARY KEY (experiment_id, rank, iter, microbatch, op_seq),
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS microbatch_metrics (
    experiment_id           TEXT,
    rank                    INTEGER,
    iter                    INTEGER,
    microbatch              INTEGER,
    compute_nonoverlap_ms   REAL,
    comm_nonoverlap_ms      REAL,
    total_union_ms          REAL,
    compute_sum_ms          REAL,
    comm_sum_ms             REAL,
    total_sum_ms            REAL,
    PRIMARY KEY (experiment_id, rank, iter, microbatch),
    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS global_load_metrics (
    experiment_id               TEXT PRIMARY KEY
        REFERENCES experiments(experiment_id),
    overall_load_start_ts       TIMESTAMP,
    overall_load_end_ts         TIMESTAMP,
    overall_load_duration_ms    REAL,
    slowest_rank                INTEGER,
    slowest_rank_load_ms        REAL,
    fastest_rank                INTEGER,
    fastest_rank_load_ms        REAL,
    overall_nccl_init_ms        REAL,
    total_memory_after_load_mb  REAL
);

CREATE TABLE IF NOT EXISTS global_step_metrics (
    experiment_id           TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    iter                    INTEGER NOT NULL,
    iter_start_ts           TIMESTAMP,
    iter_end_ts             TIMESTAMP,
    loss                    REAL,
    lr                      REAL,
    iter_duration_ms        REAL,
    total_forward_ms        REAL,
    total_backward_ms       REAL,
    total_optimizer_ms      REAL,
    total_pp_send_ms        REAL,
    total_pp_recv_ms        REAL,
    total_pp_send_cumulative_ms REAL,
    total_pp_recv_cumulative_ms REAL,
    total_pp_bytes          BIGINT,
    max_pp_send_ms          REAL,
    max_pp_recv_ms          REAL,
    max_pp_send_cumulative_ms REAL,
    max_pp_recv_cumulative_ms REAL,
    total_forward_compute_ms REAL,
    total_backward_compute_ms REAL,
    total_dp_allreduce_ms   REAL,
    max_dp_allreduce_ms     REAL,
    dp_allreduce_bytes      BIGINT,
    slowest_rank            INTEGER,
    fastest_rank            INTEGER,
    straggler_ratio         REAL,
    PRIMARY KEY (experiment_id, iter)
);

CREATE TABLE IF NOT EXISTS global_microbatch_metrics (
    experiment_id               TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    iter                        INTEGER NOT NULL,
    microbatch                  INTEGER NOT NULL,
    total_compute_nonoverlap_ms REAL,
    total_comm_nonoverlap_ms    REAL,
    total_union_ms              REAL,
    total_compute_sum_ms        REAL,
    total_comm_sum_ms           REAL,
    total_sum_ms                REAL,
    max_compute_nonoverlap_ms   REAL,
    max_comm_nonoverlap_ms      REAL,
    max_total_union_ms          REAL,
    slowest_rank                INTEGER,
    fastest_rank                INTEGER,
    straggler_ratio             REAL,
    PRIMARY KEY (experiment_id, iter, microbatch)
);

CREATE INDEX IF NOT EXISTS idx_experiments_scheduler
    ON experiments(scheduler);
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_step_metrics_exp_rank
    ON step_metrics(experiment_id, rank);
CREATE INDEX IF NOT EXISTS idx_step_metrics_duration
    ON step_metrics(duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_micro_op_exp_iter
    ON microbatch_op_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_micro_op_rank_iter
    ON microbatch_op_metrics(experiment_id, rank, iter);
CREATE INDEX IF NOT EXISTS idx_micro_metrics_exp_iter
    ON microbatch_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_micro_metrics_rank_iter
    ON microbatch_metrics(experiment_id, rank, iter);
CREATE INDEX IF NOT EXISTS idx_rank_config_mps
    ON rank_config(mps_pct);
CREATE INDEX IF NOT EXISTS idx_global_step_exp
    ON global_step_metrics(experiment_id, iter);
CREATE INDEX IF NOT EXISTS idx_global_step_straggler
    ON global_step_metrics(straggler_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_global_micro_exp_iter
    ON global_microbatch_metrics(experiment_id, iter);
""")
    conn.commit()


# ── Timestamp helpers ────────────────────────────────────


def _parse_ts(ts: str | None) -> datetime | None:
    """Parse ISO-8601 timestamp."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _duration_ms(
    start: str | None, end: str | None
) -> float | None:
    """Compute duration in ms between two ISO timestamps."""
    s = _parse_ts(start)
    e = _parse_ts(end)
    if s is None or e is None:
        return None
    return (e - s).total_seconds() * 1000.0


def _to_float(value: Any) -> float | None:
    """Convert numeric-like input to float, else None."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_union_ms(
    intervals: list[tuple[float, float]],
) -> float:
    """Union length (ms) of [start, end] intervals."""
    cleaned: list[tuple[float, float]] = []
    for start, end in intervals:
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            continue
        s = float(start)
        e = float(end)
        if e < s:
            continue
        cleaned.append((s, e))
    if not cleaned:
        return 0.0
    cleaned.sort(key=lambda x: x[0])
    cur_s, cur_e = cleaned[0]
    total = 0.0
    for s, e in cleaned[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        total += cur_e - cur_s
        cur_s, cur_e = s, e
    total += cur_e - cur_s
    return float(total)


# ── JSONL reader ─────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ── Stitcher ─────────────────────────────────────────────


def stitch_experiment(
    experiment_id: str,
    data_dir: Path,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    """Process all rank JSONL files for one experiment.

    Returns summary dict with counts.
    """
    # Collect all records by rank
    all_records: dict[int, list[dict]] = {}
    for jsonl_file in sorted(data_dir.glob("rank_*.jsonl")):
        rank_str = jsonl_file.stem.split("_")[1]
        rank = int(rank_str)
        all_records[rank] = read_jsonl(jsonl_file)

    if not all_records:
        return {"error": "No JSONL files found"}

    load_count = 0
    step_count = 0
    global_raw_count = 0
    micro_op_count = 0
    microbatch_count = 0
    global_microbatch_count = 0

    # ── Process load records ────────────────────────
    load_data: dict[int, dict] = {}
    for rank, records in all_records.items():
        for rec in records:
            if rec.get("type") != "load":
                continue
            dur = _duration_ms(
                rec.get("model_load_start_ts"),
                rec.get("model_load_end_ts"),
            )
            nccl_dur = _duration_ms(
                rec.get("nccl_init_start_ts"),
                rec.get("nccl_init_end_ts"),
            )
            conn.execute(
                """INSERT OR REPLACE INTO load_metrics
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    rank,
                    rec.get("model_load_start_ts"),
                    rec.get("model_load_end_ts"),
                    dur,
                    rec.get("nccl_init_start_ts"),
                    rec.get("nccl_init_end_ts"),
                    nccl_dur,
                    rec.get("memory_after_load_mb"),
                ),
            )
            load_data[rank] = {
                "start_ts": rec.get("model_load_start_ts"),
                "end_ts": rec.get("model_load_end_ts"),
                "duration_ms": dur,
                "nccl_duration_ms": nccl_dur,
                "memory_mb": rec.get("memory_after_load_mb"),
            }
            load_count += 1

    # ── Process step records ────────────────────────
    # Group by (rank, iter) for step_metrics
    step_cols = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(step_metrics)"
        ).fetchall()
    }
    fwd_col = (
        "forward_ms" if "forward_ms" in step_cols else "fwd_ms"
    )
    bwd_col = (
        "backward_ms" if "backward_ms" in step_cols else "bwd_ms"
    )
    opt_col = (
        "optimizer_step_ms"
        if "optimizer_step_ms" in step_cols
        else "opt_ms"
    )
    step_insert_sql = f"""INSERT OR REPLACE INTO step_metrics
                (
                    experiment_id,
                    rank,
                    iter,
                    start_ts,
                    end_ts,
                    duration_ms,
                    forward_start_ts,
                    forward_end_ts,
                    {fwd_col},
                    backward_start_ts,
                    backward_end_ts,
                    {bwd_col},
                    optimizer_start_ts,
                    optimizer_end_ts,
                    {opt_col},
                    pp_send_start_ts,
                    pp_send_end_ts,
                    pp_send_ms,
                    pp_recv_start_ts,
                    pp_recv_end_ts,
                    pp_recv_ms,
                    pp_send_cumulative_ms,
                    pp_recv_cumulative_ms,
                    forward_compute_ms,
                    backward_compute_ms,
                    pp_send_count,
                    pp_recv_count,
                    pp_send_bytes,
                    pp_recv_bytes,
                    dp_allreduce_start_ts,
                    dp_allreduce_end_ts,
                    dp_allreduce_ms,
                    dp_allreduce_bytes,
                    loss,
                    lr,
                    memory_allocated_mb,
                    memory_reserved_mb,
                    memory_peak_mb
                )
                VALUES (
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?
                )"""
    step_data: dict[int, dict[int, dict]] = {}
    for rank, records in all_records.items():
        step_data[rank] = {}
        for rec in records:
            if rec.get("type") != "step":
                continue
            it = rec.get("iter", -1)
            dur = _duration_ms(
                rec.get("start_ts"), rec.get("end_ts")
            )
            fwd_ms = _duration_ms(
                rec.get("forward_start_ts"),
                rec.get("forward_end_ts"),
            )
            bwd_ms = _duration_ms(
                rec.get("backward_start_ts"),
                rec.get("backward_end_ts"),
            )
            opt_ms = _duration_ms(
                rec.get("optimizer_start_ts"),
                rec.get("optimizer_end_ts"),
            )
            pp_send_ms = _duration_ms(
                rec.get("pp_send_start_ts"),
                rec.get("pp_send_end_ts"),
            )
            pp_recv_ms = _duration_ms(
                rec.get("pp_recv_start_ts"),
                rec.get("pp_recv_end_ts"),
            )
            pp_send_cum_ms = _to_float(
                rec.get("pp_send_cumulative_ms")
            )
            pp_recv_cum_ms = _to_float(
                rec.get("pp_recv_cumulative_ms")
            )
            fwd_compute_ms = _to_float(
                rec.get("forward_compute_ms")
            )
            bwd_compute_ms = _to_float(
                rec.get("backward_compute_ms")
            )
            dp_ar_ms = _duration_ms(
                rec.get("dp_allreduce_start_ts"),
                rec.get("dp_allreduce_end_ts"),
            )

            conn.execute(
                step_insert_sql,
                (
                    experiment_id,
                    rank,
                    it,
                    rec.get("start_ts"),
                    rec.get("end_ts"),
                    dur,
                    rec.get("forward_start_ts"),
                    rec.get("forward_end_ts"),
                    fwd_ms,
                    rec.get("backward_start_ts"),
                    rec.get("backward_end_ts"),
                    bwd_ms,
                    rec.get("optimizer_start_ts"),
                    rec.get("optimizer_end_ts"),
                    opt_ms,
                    rec.get("pp_send_start_ts"),
                    rec.get("pp_send_end_ts"),
                    pp_send_ms,
                    rec.get("pp_recv_start_ts"),
                    rec.get("pp_recv_end_ts"),
                    pp_recv_ms,
                    pp_send_cum_ms,
                    pp_recv_cum_ms,
                    fwd_compute_ms,
                    bwd_compute_ms,
                    rec.get("pp_send_count", 0),
                    rec.get("pp_recv_count", 0),
                    rec.get("pp_send_bytes", 0),
                    rec.get("pp_recv_bytes", 0),
                    rec.get("dp_allreduce_start_ts"),
                    rec.get("dp_allreduce_end_ts"),
                    dp_ar_ms,
                    rec.get("dp_allreduce_bytes", 0),
                    rec.get("loss"),
                    rec.get("lr"),
                    rec.get("memory_allocated_mb"),
                    rec.get("memory_reserved_mb"),
                    rec.get("memory_peak_mb"),
                ),
            )
            step_data[rank][it] = {
                "duration_ms": dur,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "opt_ms": opt_ms,
                "pp_send_ms": pp_send_ms,
                "pp_recv_ms": pp_recv_ms,
                "pp_send_cum_ms": pp_send_cum_ms,
                "pp_recv_cum_ms": pp_recv_cum_ms,
                "fwd_compute_ms": fwd_compute_ms,
                "bwd_compute_ms": bwd_compute_ms,
                "dp_ar_ms": dp_ar_ms,
                "pp_send_bytes": rec.get(
                    "pp_send_bytes", 0
                ),
                "dp_allreduce_bytes": rec.get(
                    "dp_allreduce_bytes", 0
                ),
            }
            step_count += 1

    # ── Process per-microbatch op timeline records ───────
    micro_ops_by_key: dict[
        tuple[int, int, int], list[dict[str, Any]]
    ] = {}
    for rank, records in all_records.items():
        for rec in records:
            if rec.get("type") != "micro_op":
                continue
            it = int(rec.get("iter", -1))
            mb = int(rec.get("microbatch", -1))
            op_seq = int(rec.get("op_seq", -1))
            op_kind = str(rec.get("op_kind", ""))
            stream = str(rec.get("stream", ""))
            start_off = _to_float(rec.get("start_offset_ms"))
            end_off = _to_float(rec.get("end_offset_ms"))
            dur_ms = _to_float(rec.get("duration_ms"))
            nbytes = int(rec.get("nbytes", 0) or 0)
            peer_rank = rec.get("peer_rank")
            if peer_rank is not None:
                try:
                    peer_rank = int(peer_rank)
                except Exception:
                    peer_rank = None

            conn.execute(
                """INSERT OR REPLACE INTO microbatch_op_metrics
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    rank,
                    it,
                    mb,
                    op_seq,
                    op_kind,
                    stream,
                    start_off,
                    end_off,
                    dur_ms,
                    nbytes,
                    peer_rank,
                ),
            )
            micro_op_count += 1
            micro_ops_by_key.setdefault(
                (rank, it, mb), []
            ).append(
                {
                    "op_kind": op_kind,
                    "start_offset_ms": start_off,
                    "end_offset_ms": end_off,
                    "duration_ms": dur_ms,
                }
            )

    compute_kinds = {"fwd_compute", "bwd_compute"}
    comm_kinds = {
        "fwd_pp_send",
        "fwd_pp_recv",
        "bwd_pp_send",
        "bwd_pp_recv",
    }
    global_micro_data: dict[
        tuple[int, int], list[dict[str, Any]]
    ] = {}
    for (rank, it, mb), ops in micro_ops_by_key.items():
        compute_intervals: list[tuple[float, float]] = []
        comm_intervals: list[tuple[float, float]] = []
        all_intervals: list[tuple[float, float]] = []
        compute_sum = 0.0
        comm_sum = 0.0

        for op in ops:
            s = _to_float(op.get("start_offset_ms"))
            e = _to_float(op.get("end_offset_ms"))
            d = _to_float(op.get("duration_ms"))
            kind = str(op.get("op_kind", ""))
            if s is not None and e is not None:
                all_intervals.append((s, e))
                if kind in compute_kinds:
                    compute_intervals.append((s, e))
                if kind in comm_kinds:
                    comm_intervals.append((s, e))
            if d is not None:
                if kind in compute_kinds:
                    compute_sum += d
                if kind in comm_kinds:
                    comm_sum += d

        compute_nonoverlap = _interval_union_ms(compute_intervals)
        comm_nonoverlap = _interval_union_ms(comm_intervals)
        total_union = _interval_union_ms(all_intervals)
        total_sum = compute_sum + comm_sum

        conn.execute(
            """INSERT OR REPLACE INTO microbatch_metrics
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                experiment_id,
                rank,
                it,
                mb,
                compute_nonoverlap,
                comm_nonoverlap,
                total_union,
                compute_sum,
                comm_sum,
                total_sum,
            ),
        )
        microbatch_count += 1
        global_micro_data.setdefault((it, mb), []).append(
            {
                "rank": rank,
                "compute_nonoverlap_ms": compute_nonoverlap,
                "comm_nonoverlap_ms": comm_nonoverlap,
                "total_union_ms": total_union,
                "compute_sum_ms": compute_sum,
                "comm_sum_ms": comm_sum,
                "total_sum_ms": total_sum,
            }
        )

    for (it, mb) in sorted(global_micro_data.keys()):
        rows = global_micro_data[(it, mb)]
        c_non = [
            float(r["compute_nonoverlap_ms"]) for r in rows
        ]
        k_non = [
            float(r["comm_nonoverlap_ms"]) for r in rows
        ]
        t_non = [float(r["total_union_ms"]) for r in rows]
        c_sum = [float(r["compute_sum_ms"]) for r in rows]
        k_sum = [float(r["comm_sum_ms"]) for r in rows]
        t_sum = [float(r["total_sum_ms"]) for r in rows]

        total_union_by_rank = {
            int(r["rank"]): float(r["total_union_ms"])
            for r in rows
        }
        slowest_rank = (
            max(
                total_union_by_rank,
                key=total_union_by_rank.get,
            )
            if total_union_by_rank
            else None
        )
        fastest_rank = (
            min(
                total_union_by_rank,
                key=total_union_by_rank.get,
            )
            if total_union_by_rank
            else None
        )
        straggler = None
        if slowest_rank is not None and fastest_rank is not None:
            fast_v = total_union_by_rank[fastest_rank]
            slow_v = total_union_by_rank[slowest_rank]
            if fast_v > 0:
                straggler = slow_v / fast_v

        conn.execute(
            """INSERT OR REPLACE INTO global_microbatch_metrics
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                experiment_id,
                it,
                mb,
                sum(c_non) if c_non else None,
                sum(k_non) if k_non else None,
                sum(t_non) if t_non else None,
                sum(c_sum) if c_sum else None,
                sum(k_sum) if k_sum else None,
                sum(t_sum) if t_sum else None,
                max(c_non) if c_non else None,
                max(k_non) if k_non else None,
                max(t_non) if t_non else None,
                slowest_rank,
                fastest_rank,
                straggler,
            ),
        )
        global_microbatch_count += 1

    # ── Compute global_load_metrics ─────────────────
    if load_data:
        starts = [
            _parse_ts(d["start_ts"])
            for d in load_data.values()
            if _parse_ts(d["start_ts"])
        ]
        ends = [
            _parse_ts(d["end_ts"])
            for d in load_data.values()
            if _parse_ts(d["end_ts"])
        ]
        durations = {
            r: d["duration_ms"]
            for r, d in load_data.items()
            if d["duration_ms"] is not None
        }
        nccl_durations = {
            r: d["nccl_duration_ms"]
            for r, d in load_data.items()
            if d["nccl_duration_ms"] is not None
        }
        memories = [
            d["memory_mb"]
            for d in load_data.values()
            if d["memory_mb"] is not None
        ]

        overall_start = (
            min(starts).isoformat() if starts else None
        )
        overall_end = (
            max(ends).isoformat() if ends else None
        )
        overall_dur = _duration_ms(
            overall_start, overall_end
        )

        slowest = (
            max(durations, key=durations.get)
            if durations else None
        )
        fastest = (
            min(durations, key=durations.get)
            if durations else None
        )

        conn.execute(
            """INSERT OR REPLACE INTO global_load_metrics
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                experiment_id,
                overall_start,
                overall_end,
                overall_dur,
                slowest,
                durations.get(slowest) if slowest is not None else None,
                fastest,
                durations.get(fastest) if fastest is not None else None,
                max(nccl_durations.values())
                if nccl_durations else None,
                sum(memories) if memories else None,
            ),
        )

    # ── Compute global_step_metrics ─────────────────
    # Collect global_step_raw records from rank 0
    global_raw: dict[int, dict] = {}
    for rec in all_records.get(0, []):
        if rec.get("type") == "global_step_raw":
            it = rec.get("iter", -1)
            global_raw[it] = rec
            global_raw_count += 1

    # Find all iterations across all ranks
    all_iters: set[int] = set()
    for rank_steps in step_data.values():
        all_iters.update(rank_steps.keys())

    for it in sorted(all_iters):
        # Raw values from rank 0
        raw = global_raw.get(it, {})

        # Cross-rank aggregation
        rank_durations: dict[int, float] = {}
        total_fwd = 0.0
        total_bwd = 0.0
        total_opt = 0.0
        total_pp_send = 0.0
        total_pp_recv = 0.0
        total_pp_send_cum = 0.0
        total_pp_recv_cum = 0.0
        has_pp_send_cum = False
        has_pp_recv_cum = False
        total_pp_bytes = 0
        max_pp_send = 0.0
        max_pp_recv = 0.0
        max_pp_send_cum = 0.0
        max_pp_recv_cum = 0.0
        has_max_pp_send_cum = False
        has_max_pp_recv_cum = False
        total_fwd_compute = 0.0
        total_bwd_compute = 0.0
        has_fwd_compute = False
        has_bwd_compute = False
        total_dp_ar = 0.0
        max_dp_ar = 0.0
        dp_ar_bytes = 0

        for rank, rank_steps in step_data.items():
            if it not in rank_steps:
                continue
            sd = rank_steps[it]
            d = sd.get("duration_ms")
            if d is not None:
                rank_durations[rank] = d
            if sd.get("fwd_ms") is not None:
                total_fwd += sd["fwd_ms"]
            if sd.get("bwd_ms") is not None:
                total_bwd += sd["bwd_ms"]
            if sd.get("opt_ms") is not None:
                total_opt += sd["opt_ms"]
            if sd.get("pp_send_ms") is not None:
                total_pp_send += sd["pp_send_ms"]
                max_pp_send = max(
                    max_pp_send, sd["pp_send_ms"]
                )
            if sd.get("pp_recv_ms") is not None:
                total_pp_recv += sd["pp_recv_ms"]
                max_pp_recv = max(
                    max_pp_recv, sd["pp_recv_ms"]
                )
            if sd.get("pp_send_cum_ms") is not None:
                has_pp_send_cum = True
                has_max_pp_send_cum = True
                total_pp_send_cum += sd["pp_send_cum_ms"]
                max_pp_send_cum = max(
                    max_pp_send_cum, sd["pp_send_cum_ms"]
                )
            if sd.get("pp_recv_cum_ms") is not None:
                has_pp_recv_cum = True
                has_max_pp_recv_cum = True
                total_pp_recv_cum += sd["pp_recv_cum_ms"]
                max_pp_recv_cum = max(
                    max_pp_recv_cum, sd["pp_recv_cum_ms"]
                )
            if sd.get("fwd_compute_ms") is not None:
                has_fwd_compute = True
                total_fwd_compute += sd["fwd_compute_ms"]
            if sd.get("bwd_compute_ms") is not None:
                has_bwd_compute = True
                total_bwd_compute += sd["bwd_compute_ms"]
            total_pp_bytes += sd.get(
                "pp_send_bytes", 0
            )
            if sd.get("dp_ar_ms") is not None:
                total_dp_ar += sd["dp_ar_ms"]
                max_dp_ar = max(
                    max_dp_ar, sd["dp_ar_ms"]
                )
            dp_ar_bytes = max(
                dp_ar_bytes,
                sd.get("dp_allreduce_bytes", 0),
            )

        # Iter duration = max(end_ts) - min(start_ts)
        # across ranks. Use duration_ms from step_data
        # for simplicity (max duration among ranks).
        iter_dur = (
            max(rank_durations.values())
            if rank_durations else None
        )

        slowest_rank = (
            max(rank_durations, key=rank_durations.get)
            if rank_durations else None
        )
        fastest_rank = (
            min(rank_durations, key=rank_durations.get)
            if rank_durations else None
        )
        straggler = None
        if rank_durations and len(rank_durations) > 1:
            min_d = min(rank_durations.values())
            max_d = max(rank_durations.values())
            if min_d > 0:
                straggler = max_d / min_d

        conn.execute(
            """INSERT OR REPLACE INTO global_step_metrics
            (
                experiment_id,
                iter,
                iter_start_ts,
                iter_end_ts,
                loss,
                lr,
                iter_duration_ms,
                total_forward_ms,
                total_backward_ms,
                total_optimizer_ms,
                total_pp_send_ms,
                total_pp_recv_ms,
                total_pp_send_cumulative_ms,
                total_pp_recv_cumulative_ms,
                total_pp_bytes,
                max_pp_send_ms,
                max_pp_recv_ms,
                max_pp_send_cumulative_ms,
                max_pp_recv_cumulative_ms,
                total_forward_compute_ms,
                total_backward_compute_ms,
                total_dp_allreduce_ms,
                max_dp_allreduce_ms,
                dp_allreduce_bytes,
                slowest_rank,
                fastest_rank,
                straggler_ratio
            )
            VALUES (
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?,?,?,?,
                ?,?,?,?,?,?,?
            )""",
            (
                experiment_id,
                it,
                raw.get("iter_start_ts"),
                raw.get("iter_end_ts"),
                raw.get("loss"),
                raw.get("lr"),
                iter_dur,
                total_fwd or None,
                total_bwd or None,
                total_opt or None,
                total_pp_send or None,
                total_pp_recv or None,
                (
                    total_pp_send_cum
                    if has_pp_send_cum
                    else None
                ),
                (
                    total_pp_recv_cum
                    if has_pp_recv_cum
                    else None
                ),
                total_pp_bytes or None,
                max_pp_send or None,
                max_pp_recv or None,
                (
                    max_pp_send_cum
                    if has_max_pp_send_cum
                    else None
                ),
                (
                    max_pp_recv_cum
                    if has_max_pp_recv_cum
                    else None
                ),
                (
                    total_fwd_compute
                    if has_fwd_compute
                    else None
                ),
                (
                    total_bwd_compute
                    if has_bwd_compute
                    else None
                ),
                total_dp_ar or None,
                max_dp_ar or None,
                dp_ar_bytes or None,
                slowest_rank,
                fastest_rank,
                straggler,
            ),
        )

    conn.commit()

    return {
        "ranks": len(all_records),
        "load_records": load_count,
        "step_records": step_count,
        "micro_op_records": micro_op_count,
        "microbatch_records": microbatch_count,
        "global_microbatch_records": global_microbatch_count,
        "global_raw_records": global_raw_count,
        "iterations": len(all_iters),
    }


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stitch JSONL metrics into SQLite",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="SQLite database path",
    )
    group = parser.add_mutually_exclusive_group(
        required=True
    )
    group.add_argument(
        "--experiment-id",
        help="Single experiment ID to stitch",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Stitch all experiments in output dir",
    )
    parser.add_argument(
        "--data-dir",
        help="Data dir for single experiment",
    )
    parser.add_argument(
        "--output-dir",
        help="Parent dir for --all mode",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)

    if args.all:
        if not args.output_dir:
            print(
                "ERROR: --output-dir required with --all",
                file=sys.stderr,
            )
            sys.exit(1)

        output_dir = Path(args.output_dir)
        total = 0
        for exp_dir in sorted(output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if not any(exp_dir.glob("rank_*.jsonl")):
                continue
            eid = exp_dir.name
            print(f"Stitching {eid}...", end=" ")
            result = stitch_experiment(
                eid, exp_dir, conn
            )
            print(json.dumps(result))
            total += 1
        print(f"\nStitched {total} experiments.")
    else:
        if not args.data_dir:
            print(
                "ERROR: --data-dir required with"
                " --experiment-id",
                file=sys.stderr,
            )
            sys.exit(1)

        data_dir = Path(args.data_dir)
        result = stitch_experiment(
            args.experiment_id, data_dir, conn
        )
        print(json.dumps(result, indent=2))

    conn.close()


if __name__ == "__main__":
    main()
