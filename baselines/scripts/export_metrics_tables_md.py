#!/usr/bin/env python3
"""Export stitched experiment metrics from SQLite into Markdown tables.

Generates a report similar to the manually curated sample:
- Global iteration metrics per experiment
- Node-level metrics for selected ranks per experiment
- Mean/std rows per table

Includes both legacy envelope timing fields and new cumulative/compute fields.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import statistics
from pathlib import Path
from typing import Any


def _table_cols(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }


def _fmt_num(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "-"


def _fmt_ts(v: Any) -> str:
    if v is None:
        return "-"
    return str(v)


def _mean_std(rows: list[dict[str, Any]], key: str) -> tuple[str, str]:
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return "-", "-"
    mean_v = statistics.fmean(vals)
    std_v = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return _fmt_num(mean_v), _fmt_num(std_v)


def _resolve_experiments(
    conn: sqlite3.Connection,
    schedulers: list[str],
    explicit_ids: list[str],
) -> list[str]:
    if explicit_ids:
        return explicit_ids

    out: list[str] = []
    for sched in schedulers:
        row = conn.execute(
            """
            SELECT experiment_id
            FROM experiments
            WHERE scheduler = ?
              AND status = 'completed'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (sched,),
        ).fetchone()
        if row:
            out.append(str(row[0]))
    return out


def _get_scheduler(conn: sqlite3.Connection, exp_id: str) -> str:
    row = conn.execute(
        "SELECT scheduler FROM experiments WHERE experiment_id=?",
        (exp_id,),
    ).fetchone()
    return str(row[0]) if row else "-"


def _get_mps_config(conn: sqlite3.Connection, exp_id: str) -> str:
    rows = conn.execute(
        """
        SELECT rank, mps_pct
        FROM rank_config
        WHERE experiment_id=?
        ORDER BY rank
        """,
        (exp_id,),
    ).fetchall()
    if not rows:
        return "-"
    return ",".join(str(int(r[1])) for r in rows)


def _get_rank_mps(conn: sqlite3.Connection, exp_id: str, rank: int) -> str:
    row = conn.execute(
        """
        SELECT mps_pct
        FROM rank_config
        WHERE experiment_id=? AND rank=?
        """,
        (exp_id, rank),
    ).fetchone()
    if not row:
        return "-"
    return str(int(row[0]))


def _fetch_global_rows(
    conn: sqlite3.Connection,
    exp_id: str,
    cols: set[str],
) -> list[dict[str, Any]]:
    wanted = [
        "iter",
        "iter_start_ts",
        "iter_end_ts",
        "iter_duration_ms",
        "total_forward_ms",
        "total_backward_ms",
        "total_optimizer_ms",
        "total_dp_allreduce_ms",
        "total_pp_send_ms",
        "total_pp_recv_ms",
        "total_pp_send_cumulative_ms",
        "total_pp_recv_cumulative_ms",
        "max_pp_send_cumulative_ms",
        "max_pp_recv_cumulative_ms",
        "total_forward_compute_ms",
        "total_backward_compute_ms",
    ]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        return []
    query = (
        "SELECT " + ", ".join(select_cols) +
        " FROM global_step_metrics WHERE experiment_id=? ORDER BY iter"
    )
    cur = conn.execute(query, (exp_id,))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _fetch_node_rows(
    conn: sqlite3.Connection,
    exp_id: str,
    rank: int,
    cols: set[str],
) -> list[dict[str, Any]]:
    fwd_col = "forward_ms" if "forward_ms" in cols else "fwd_ms"
    bwd_col = "backward_ms" if "backward_ms" in cols else "bwd_ms"
    opt_col = (
        "optimizer_step_ms"
        if "optimizer_step_ms" in cols
        else "opt_ms"
    )
    wanted = [
        "iter",
        "start_ts",
        "end_ts",
        "duration_ms",
        fwd_col,
        bwd_col,
        opt_col,
        "dp_allreduce_ms",
        "pp_send_ms",
        "pp_recv_ms",
        "pp_send_cumulative_ms",
        "pp_recv_cumulative_ms",
        "forward_compute_ms",
        "backward_compute_ms",
    ]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        return []
    query = (
        "SELECT " + ", ".join(select_cols) +
        " FROM step_metrics WHERE experiment_id=? AND rank=? ORDER BY iter"
    )
    cur = conn.execute(query, (exp_id, rank))
    names = [d[0] for d in cur.description]
    rows = [dict(zip(names, row)) for row in cur.fetchall()]
    for r in rows:
        r["forward_ms"] = r.get(fwd_col)
        r["backward_ms"] = r.get(bwd_col)
        r["optimizer_step_ms"] = r.get(opt_col)
    return rows


def _fetch_global_microbatch_rows(
    conn: sqlite3.Connection,
    exp_id: str,
    cols: set[str],
) -> list[dict[str, Any]]:
    wanted = [
        "iter",
        "microbatch",
        "total_compute_nonoverlap_ms",
        "total_comm_nonoverlap_ms",
        "total_union_ms",
        "total_compute_sum_ms",
        "total_comm_sum_ms",
        "total_sum_ms",
        "max_compute_nonoverlap_ms",
        "max_comm_nonoverlap_ms",
        "max_total_union_ms",
        "slowest_rank",
        "fastest_rank",
        "straggler_ratio",
    ]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        return []
    query = (
        "SELECT " + ", ".join(select_cols) +
        " FROM global_microbatch_metrics WHERE experiment_id=? "
        "ORDER BY iter, microbatch"
    )
    cur = conn.execute(query, (exp_id,))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _fetch_rank_microbatch_rows(
    conn: sqlite3.Connection,
    exp_id: str,
    rank: int,
    cols: set[str],
) -> list[dict[str, Any]]:
    wanted = [
        "iter",
        "microbatch",
        "compute_nonoverlap_ms",
        "comm_nonoverlap_ms",
        "total_union_ms",
        "compute_sum_ms",
        "comm_sum_ms",
        "total_sum_ms",
    ]
    select_cols = [c for c in wanted if c in cols]
    if not select_cols:
        return []
    query = (
        "SELECT " + ", ".join(select_cols) +
        " FROM microbatch_metrics WHERE experiment_id=? AND rank=? "
        "ORDER BY iter, microbatch"
    )
    cur = conn.execute(query, (exp_id, rank))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _append_global_table(
    lines: list[str],
    exp_id: str,
    scheduler: str,
    mps_cfg: str,
    rows: list[dict[str, Any]],
) -> None:
    lines.append(f"## Global Iteration Metrics — {exp_id}")
    lines.append(
        f"Scheduler: {scheduler} | MPS config (R0..R7): {mps_cfg}"
    )
    lines.append("")
    lines.append(
        "| Iter | Global Iter Start (UTC) | Global Iter End (UTC) | "
        "Iter Duration (ms) | total_forward_ms | total_backward_ms | "
        "total_optimizer_ms | total_dp_allreduce_ms | total_pp_send_ms | "
        "total_pp_recv_ms | total_pp_send_cumulative_ms | "
        "total_pp_recv_cumulative_ms | max_pp_send_cumulative_ms | "
        "max_pp_recv_cumulative_ms | total_forward_compute_ms | "
        "total_backward_compute_ms | Scheduler |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---:|---|"
    )
    for r in rows:
        lines.append(
            f"| {r.get('iter', '-')} | {_fmt_ts(r.get('iter_start_ts'))} | "
            f"{_fmt_ts(r.get('iter_end_ts'))} | {_fmt_num(r.get('iter_duration_ms'))} | "
            f"{_fmt_num(r.get('total_forward_ms'))} | {_fmt_num(r.get('total_backward_ms'))} | "
            f"{_fmt_num(r.get('total_optimizer_ms'))} | {_fmt_num(r.get('total_dp_allreduce_ms'))} | "
            f"{_fmt_num(r.get('total_pp_send_ms'))} | {_fmt_num(r.get('total_pp_recv_ms'))} | "
            f"{_fmt_num(r.get('total_pp_send_cumulative_ms'))} | {_fmt_num(r.get('total_pp_recv_cumulative_ms'))} | "
            f"{_fmt_num(r.get('max_pp_send_cumulative_ms'))} | {_fmt_num(r.get('max_pp_recv_cumulative_ms'))} | "
            f"{_fmt_num(r.get('total_forward_compute_ms'))} | {_fmt_num(r.get('total_backward_compute_ms'))} | "
            f"{scheduler} |"
        )

    avg = {
        "iter_duration_ms": _mean_std(rows, "iter_duration_ms")[0],
        "total_forward_ms": _mean_std(rows, "total_forward_ms")[0],
        "total_backward_ms": _mean_std(rows, "total_backward_ms")[0],
        "total_optimizer_ms": _mean_std(rows, "total_optimizer_ms")[0],
        "total_dp_allreduce_ms": _mean_std(rows, "total_dp_allreduce_ms")[0],
        "total_pp_send_ms": _mean_std(rows, "total_pp_send_ms")[0],
        "total_pp_recv_ms": _mean_std(rows, "total_pp_recv_ms")[0],
        "total_pp_send_cumulative_ms": _mean_std(rows, "total_pp_send_cumulative_ms")[0],
        "total_pp_recv_cumulative_ms": _mean_std(rows, "total_pp_recv_cumulative_ms")[0],
        "max_pp_send_cumulative_ms": _mean_std(rows, "max_pp_send_cumulative_ms")[0],
        "max_pp_recv_cumulative_ms": _mean_std(rows, "max_pp_recv_cumulative_ms")[0],
        "total_forward_compute_ms": _mean_std(rows, "total_forward_compute_ms")[0],
        "total_backward_compute_ms": _mean_std(rows, "total_backward_compute_ms")[0],
    }
    std = {
        "iter_duration_ms": _mean_std(rows, "iter_duration_ms")[1],
        "total_forward_ms": _mean_std(rows, "total_forward_ms")[1],
        "total_backward_ms": _mean_std(rows, "total_backward_ms")[1],
        "total_optimizer_ms": _mean_std(rows, "total_optimizer_ms")[1],
        "total_dp_allreduce_ms": _mean_std(rows, "total_dp_allreduce_ms")[1],
        "total_pp_send_ms": _mean_std(rows, "total_pp_send_ms")[1],
        "total_pp_recv_ms": _mean_std(rows, "total_pp_recv_ms")[1],
        "total_pp_send_cumulative_ms": _mean_std(rows, "total_pp_send_cumulative_ms")[1],
        "total_pp_recv_cumulative_ms": _mean_std(rows, "total_pp_recv_cumulative_ms")[1],
        "max_pp_send_cumulative_ms": _mean_std(rows, "max_pp_send_cumulative_ms")[1],
        "max_pp_recv_cumulative_ms": _mean_std(rows, "max_pp_recv_cumulative_ms")[1],
        "total_forward_compute_ms": _mean_std(rows, "total_forward_compute_ms")[1],
        "total_backward_compute_ms": _mean_std(rows, "total_backward_compute_ms")[1],
    }

    lines.append(
        f"| Avg |  |  | {avg['iter_duration_ms']} | {avg['total_forward_ms']} | "
        f"{avg['total_backward_ms']} | {avg['total_optimizer_ms']} | "
        f"{avg['total_dp_allreduce_ms']} | {avg['total_pp_send_ms']} | "
        f"{avg['total_pp_recv_ms']} | {avg['total_pp_send_cumulative_ms']} | "
        f"{avg['total_pp_recv_cumulative_ms']} | {avg['max_pp_send_cumulative_ms']} | "
        f"{avg['max_pp_recv_cumulative_ms']} | {avg['total_forward_compute_ms']} | "
        f"{avg['total_backward_compute_ms']} |  |"
    )
    lines.append(
        f"| Std |  |  | {std['iter_duration_ms']} | {std['total_forward_ms']} | "
        f"{std['total_backward_ms']} | {std['total_optimizer_ms']} | "
        f"{std['total_dp_allreduce_ms']} | {std['total_pp_send_ms']} | "
        f"{std['total_pp_recv_ms']} | {std['total_pp_send_cumulative_ms']} | "
        f"{std['total_pp_recv_cumulative_ms']} | {std['max_pp_send_cumulative_ms']} | "
        f"{std['max_pp_recv_cumulative_ms']} | {std['total_forward_compute_ms']} | "
        f"{std['total_backward_compute_ms']} |  |"
    )
    lines.append("")


def _append_global_microbatch_table(
    lines: list[str],
    exp_id: str,
    scheduler: str,
    rows: list[dict[str, Any]],
) -> None:
    lines.append(f"## Global Microbatch Metrics — {exp_id}")
    lines.append(f"Scheduler: {scheduler}")
    lines.append("")
    lines.append(
        "| Iter | Microbatch | total_compute_nonoverlap_ms | "
        "total_comm_nonoverlap_ms | total_union_ms | "
        "total_compute_sum_ms | total_comm_sum_ms | total_sum_ms | "
        "max_compute_nonoverlap_ms | max_comm_nonoverlap_ms | "
        "max_total_union_ms | slowest_rank | fastest_rank | straggler_ratio |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            f"| {r.get('iter', '-')} | {r.get('microbatch', '-')} | "
            f"{_fmt_num(r.get('total_compute_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('total_comm_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('total_union_ms'))} | "
            f"{_fmt_num(r.get('total_compute_sum_ms'))} | "
            f"{_fmt_num(r.get('total_comm_sum_ms'))} | "
            f"{_fmt_num(r.get('total_sum_ms'))} | "
            f"{_fmt_num(r.get('max_compute_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('max_comm_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('max_total_union_ms'))} | "
            f"{r.get('slowest_rank', '-')} | "
            f"{r.get('fastest_rank', '-')} | "
            f"{_fmt_num(r.get('straggler_ratio'))} |"
        )
    lines.append("")


def _append_node_table(
    lines: list[str],
    exp_id: str,
    scheduler: str,
    mps_cfg: str,
    rank: int,
    rank_mps: str,
    rows: list[dict[str, Any]],
) -> None:
    lines.append(f"## Node-Level Metrics — Experiment {exp_id}, Rank {rank}")
    lines.append(
        f"Scheduler: {scheduler} | Rank {rank} MPS: {rank_mps}% | "
        f"Experiment MPS config (R0..R7): {mps_cfg}"
    )
    lines.append("")
    lines.append(
        "| Iter | Node Iter Start (UTC) | Node Iter End (UTC) | "
        "Step Time (ms) | forward_ms | backward_ms | optimizer_step_ms | "
        "dp_allreduce_ms | pp_send_ms | pp_recv_ms | "
        "pp_send_cumulative_ms | pp_recv_cumulative_ms | "
        "forward_compute_ms | backward_compute_ms | Rank MPS (%) | Scheduler |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---:|---|"
    )
    for r in rows:
        lines.append(
            f"| {r.get('iter', '-')} | {_fmt_ts(r.get('start_ts'))} | "
            f"{_fmt_ts(r.get('end_ts'))} | {_fmt_num(r.get('duration_ms'))} | "
            f"{_fmt_num(r.get('forward_ms'))} | {_fmt_num(r.get('backward_ms'))} | "
            f"{_fmt_num(r.get('optimizer_step_ms'))} | {_fmt_num(r.get('dp_allreduce_ms'))} | "
            f"{_fmt_num(r.get('pp_send_ms'))} | {_fmt_num(r.get('pp_recv_ms'))} | "
            f"{_fmt_num(r.get('pp_send_cumulative_ms'))} | {_fmt_num(r.get('pp_recv_cumulative_ms'))} | "
            f"{_fmt_num(r.get('forward_compute_ms'))} | {_fmt_num(r.get('backward_compute_ms'))} | "
            f"{rank_mps} | {scheduler} |"
        )

    avg = {
        "duration_ms": _mean_std(rows, "duration_ms")[0],
        "forward_ms": _mean_std(rows, "forward_ms")[0],
        "backward_ms": _mean_std(rows, "backward_ms")[0],
        "optimizer_step_ms": _mean_std(rows, "optimizer_step_ms")[0],
        "dp_allreduce_ms": _mean_std(rows, "dp_allreduce_ms")[0],
        "pp_send_ms": _mean_std(rows, "pp_send_ms")[0],
        "pp_recv_ms": _mean_std(rows, "pp_recv_ms")[0],
        "pp_send_cumulative_ms": _mean_std(rows, "pp_send_cumulative_ms")[0],
        "pp_recv_cumulative_ms": _mean_std(rows, "pp_recv_cumulative_ms")[0],
        "forward_compute_ms": _mean_std(rows, "forward_compute_ms")[0],
        "backward_compute_ms": _mean_std(rows, "backward_compute_ms")[0],
    }
    std = {
        "duration_ms": _mean_std(rows, "duration_ms")[1],
        "forward_ms": _mean_std(rows, "forward_ms")[1],
        "backward_ms": _mean_std(rows, "backward_ms")[1],
        "optimizer_step_ms": _mean_std(rows, "optimizer_step_ms")[1],
        "dp_allreduce_ms": _mean_std(rows, "dp_allreduce_ms")[1],
        "pp_send_ms": _mean_std(rows, "pp_send_ms")[1],
        "pp_recv_ms": _mean_std(rows, "pp_recv_ms")[1],
        "pp_send_cumulative_ms": _mean_std(rows, "pp_send_cumulative_ms")[1],
        "pp_recv_cumulative_ms": _mean_std(rows, "pp_recv_cumulative_ms")[1],
        "forward_compute_ms": _mean_std(rows, "forward_compute_ms")[1],
        "backward_compute_ms": _mean_std(rows, "backward_compute_ms")[1],
    }

    lines.append(
        f"| Avg |  |  | {avg['duration_ms']} | {avg['forward_ms']} | "
        f"{avg['backward_ms']} | {avg['optimizer_step_ms']} | "
        f"{avg['dp_allreduce_ms']} | {avg['pp_send_ms']} | "
        f"{avg['pp_recv_ms']} | {avg['pp_send_cumulative_ms']} | "
        f"{avg['pp_recv_cumulative_ms']} | {avg['forward_compute_ms']} | "
        f"{avg['backward_compute_ms']} |  |  |"
    )
    lines.append(
        f"| Std |  |  | {std['duration_ms']} | {std['forward_ms']} | "
        f"{std['backward_ms']} | {std['optimizer_step_ms']} | "
        f"{std['dp_allreduce_ms']} | {std['pp_send_ms']} | "
        f"{std['pp_recv_ms']} | {std['pp_send_cumulative_ms']} | "
        f"{std['pp_recv_cumulative_ms']} | {std['forward_compute_ms']} | "
        f"{std['backward_compute_ms']} |  |  |"
    )
    lines.append("")


def _append_rank_microbatch_table(
    lines: list[str],
    exp_id: str,
    scheduler: str,
    rank: int,
    rows: list[dict[str, Any]],
) -> None:
    lines.append(
        f"## Rank Microbatch Metrics — Experiment {exp_id}, Rank {rank}"
    )
    lines.append(f"Scheduler: {scheduler}")
    lines.append("")
    lines.append(
        "| Iter | Microbatch | compute_nonoverlap_ms | "
        "comm_nonoverlap_ms | total_union_ms | "
        "compute_sum_ms | comm_sum_ms | total_sum_ms |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        lines.append(
            f"| {r.get('iter', '-')} | {r.get('microbatch', '-')} | "
            f"{_fmt_num(r.get('compute_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('comm_nonoverlap_ms'))} | "
            f"{_fmt_num(r.get('total_union_ms'))} | "
            f"{_fmt_num(r.get('compute_sum_ms'))} | "
            f"{_fmt_num(r.get('comm_sum_ms'))} | "
            f"{_fmt_num(r.get('total_sum_ms'))} |"
        )
    lines.append("")


def generate_report(
    db_path: Path,
    out_path: Path,
    schedulers: list[str],
    experiment_ids: list[str],
    node_ranks: list[int],
    include_microbatch: bool = False,
) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    global_cols = _table_cols(conn, "global_step_metrics")
    step_cols = _table_cols(conn, "step_metrics")
    micro_cols = _table_cols(conn, "microbatch_metrics")
    global_micro_cols = _table_cols(
        conn, "global_microbatch_metrics"
    )

    selected = _resolve_experiments(
        conn,
        schedulers=schedulers,
        explicit_ids=experiment_ids,
    )
    if not selected:
        raise RuntimeError(
            "No experiments selected. "
            "Provide --experiment-ids or ensure completed runs exist."
        )

    lines: list[str] = []
    lines.append("# Latest Metrics Tables")
    lines.append("")
    lines.append(
        f"- Generated: `{dt.datetime.now().isoformat(sep=' ', timespec='seconds')}`"
    )
    lines.append(f"- DB: `{db_path}`")
    lines.append(
        f"- Experiments: `{', '.join(selected)}`"
    )
    lines.append(
        f"- Node ranks: `{', '.join(str(r) for r in node_ranks)}`"
    )
    lines.append("")

    for exp_id in selected:
        scheduler = _get_scheduler(conn, exp_id)
        mps_cfg = _get_mps_config(conn, exp_id)
        global_rows = _fetch_global_rows(conn, exp_id, global_cols)
        if global_rows:
            _append_global_table(
                lines, exp_id, scheduler, mps_cfg, global_rows
            )

        for rank in node_ranks:
            node_rows = _fetch_node_rows(
                conn, exp_id, rank, step_cols
            )
            if not node_rows:
                continue
            rank_mps = _get_rank_mps(conn, exp_id, rank)
            _append_node_table(
                lines,
                exp_id,
                scheduler,
                mps_cfg,
                rank,
                rank_mps,
                node_rows,
            )
            if include_microbatch:
                rank_micro = _fetch_rank_microbatch_rows(
                    conn, exp_id, rank, micro_cols
                )
                if rank_micro:
                    _append_rank_microbatch_table(
                        lines,
                        exp_id,
                        scheduler,
                        rank,
                        rank_micro,
                    )
        if include_microbatch:
            global_micro = _fetch_global_microbatch_rows(
                conn, exp_id, global_micro_cols
            )
            if global_micro:
                _append_global_microbatch_table(
                    lines,
                    exp_id,
                    scheduler,
                    global_micro,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    conn.close()
    return len(selected)


def _parse_csv_str(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_csv_int(raw: str) -> list[int]:
    out: list[int] = []
    for tok in _parse_csv_str(raw):
        out.append(int(tok))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export global/node metrics tables from experiments.db "
            "to a Markdown report."
        )
    )
    parser.add_argument(
        "--db",
        default="baselines/experiments.db",
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output",
        default=(
            "baselines/experiment_data/reports/"
            "latest_metrics_tables.md"
        ),
        help="Output markdown path.",
    )
    parser.add_argument(
        "--schedulers",
        default="asteroid,dtfm,confident",
        help=(
            "Comma-separated schedulers used when "
            "--experiment-ids is not provided."
        ),
    )
    parser.add_argument(
        "--experiment-ids",
        default="",
        help=(
            "Optional comma-separated experiment IDs. "
            "If set, overrides --schedulers selection."
        ),
    )
    parser.add_argument(
        "--node-ranks",
        default="0,3",
        help="Comma-separated ranks for node-level tables.",
    )
    parser.add_argument(
        "--include-microbatch",
        action="store_true",
        help=(
            "Include per-rank microbatch and global microbatch "
            "summary tables when available."
        ),
    )
    args = parser.parse_args()

    schedulers = _parse_csv_str(args.schedulers)
    experiment_ids = _parse_csv_str(args.experiment_ids)
    node_ranks = _parse_csv_int(args.node_ranks)

    count = generate_report(
        db_path=Path(args.db),
        out_path=Path(args.output),
        schedulers=schedulers,
        experiment_ids=experiment_ids,
        node_ranks=node_ranks,
        include_microbatch=args.include_microbatch,
    )
    print(
        f"Saved metrics report for {count} experiment(s): "
        f"{args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
