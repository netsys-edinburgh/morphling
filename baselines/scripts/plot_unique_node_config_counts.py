#!/usr/bin/env python3
"""Plot unique planner node-configuration counts per scheduler.

Definition used for uniqueness (nodes treated as distinct):
- PP signature: exact node IDs per stage, preserving stage id.
- DP signature: exact node IDs per DP slot, preserving slot id.
- Layer signature: number of model layers assigned to each stage.

A run's configuration signature is:
    (
      tuple((stage_id, (node_id, ...)), ...),
      tuple((dp_slot, (node_id, ...)), ...),
      tuple(layers_per_stage),
    )
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any


BASELINES_DIR = Path(__file__).resolve().parent.parent


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row and row[0])


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _signature_from_plan(
    plan: dict[str, Any],
    num_layers: int,
) -> tuple[
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[tuple[int, tuple[int, ...]], ...],
    tuple[int, ...],
]:
    raw_groups = plan.get("device_groups", {})
    if not isinstance(raw_groups, dict):
        raise ValueError("plan.device_groups missing or invalid")

    stage_rows: list[tuple[int, list[int]]] = []
    for stage_raw, ranks_raw in raw_groups.items():
        stage_id = _to_int(stage_raw)
        if stage_id is None or not isinstance(ranks_raw, list):
            continue
        ranks: list[int] = []
        for r in ranks_raw:
            rr = _to_int(r)
            if rr is not None:
                ranks.append(rr)
        stage_rows.append((stage_id, ranks))

    if not stage_rows:
        raise ValueError("plan.device_groups has no valid stages")

    stage_rows.sort(key=lambda x: x[0])

    # Preserve exact node ids inside each PP stage.
    pp_sig: tuple[tuple[int, tuple[int, ...]], ...] = tuple(
        (stage_id, tuple(ranks))
        for stage_id, ranks in stage_rows
    )

    # Build DP slots by taking the same position across stages.
    max_slots = max(len(ranks) for _, ranks in stage_rows)
    dp_rows: list[tuple[int, tuple[int, ...]]] = []
    for slot in range(max_slots):
        slot_nodes = tuple(
            ranks[slot]
            for _, ranks in stage_rows
            if slot < len(ranks)
        )
        if slot_nodes:
            dp_rows.append((slot, slot_nodes))
    dp_sig = tuple(dp_rows)

    # partition_points are exclusive stage boundaries.
    raw_pts = plan.get("partition_points", [])
    points: list[int] = []
    if isinstance(raw_pts, list):
        for p in raw_pts:
            pp = _to_int(p)
            if pp is None:
                continue
            if 0 < pp < num_layers:
                points.append(pp)
    points = sorted(set(points))
    bounds = [0] + points + [num_layers]
    layer_sig = tuple(
        bounds[i + 1] - bounds[i]
        for i in range(len(bounds) - 1)
    )

    return pp_sig, dp_sig, layer_sig


def load_unique_counts(
    db_path: Path,
    completed_only: bool = True,
) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        if not _table_exists(conn, "planner_results"):
            raise RuntimeError(
                "Missing table planner_results. "
                "Run planner sweep first."
            )
        if not _table_exists(conn, "experiments"):
            raise RuntimeError("Missing table experiments.")

        sql = """
            SELECT e.scheduler, e.num_layers, pr.plan_json
            FROM experiments e
            JOIN planner_results pr
              ON pr.experiment_id = e.experiment_id
        """
        params: list[Any] = []
        if completed_only:
            sql += " WHERE e.status = ?"
            params.append("completed")

        per_sched: dict[str, set[Any]] = defaultdict(set)
        cur = conn.execute(sql, params)
        for scheduler, num_layers, plan_json in cur.fetchall():
            if not scheduler or not plan_json:
                continue
            try:
                plan = json.loads(plan_json)
                sig = _signature_from_plan(
                    plan, int(num_layers)
                )
                per_sched[str(scheduler)].add(sig)
            except Exception:
                # Skip malformed plans.
                continue

        counts = {k: len(v) for k, v in per_sched.items()}
        return counts
    finally:
        conn.close()


def _scheduler_order(counts: dict[str, int]) -> tuple[list[str], list[int]]:
    if not counts:
        raise RuntimeError("No planner rows found to plot.")

    order = ["asteroid", "confident", "dtfm"]
    schedulers = [s for s in order if s in counts] + [
        s for s in sorted(counts) if s not in order
    ]
    values = [counts[s] for s in schedulers]
    return schedulers, values


def _plot_counts_matplotlib(counts: dict[str, int], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    schedulers, values = _scheduler_order(counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(schedulers, values)
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Unique Node Configurations")
    ax.set_title(
        "Unique PP+DP+Layer-Split Configurations by Scheduler"
    )
    ax.set_ylim(0, max(values) * 1.15 + 1)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(v),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_counts_svg(counts: dict[str, int], out_path: Path) -> Path:
    """Pure-SVG fallback when matplotlib is unavailable."""
    schedulers, values = _scheduler_order(counts)
    target = out_path
    if target.suffix.lower() != ".svg":
        target = target.with_suffix(".svg")

    width = 920
    height = 560
    left = 90
    right = 30
    top = 70
    bottom = 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(values) if values else 1
    y_max = max(y_max, 1)

    n = len(schedulers)
    step = plot_w / n
    bar_w = step * 0.55

    palette = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')

    # Title
    parts.append(
        '<text x="460" y="34" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#111">'
        "Unique PP+DP+Layer-Split Configurations by Scheduler"
        "</text>"
    )

    # Axes
    x0 = left
    y0 = top + plot_h
    x1 = left + plot_w
    y1 = top
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="2"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="2"/>'
    )

    # Y ticks (5)
    ticks = 5
    for i in range(ticks + 1):
        val = int(round(y_max * i / ticks))
        yy = y0 - (plot_h * i / ticks)
        parts.append(
            f'<line x1="{x0-6}" y1="{yy:.1f}" x2="{x0}" y2="{yy:.1f}" stroke="#222" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x0-10}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#333">{val}</text>'
        )

    # Bars + labels
    for i, (sched, val) in enumerate(zip(schedulers, values)):
        cx = left + (i + 0.5) * step
        bh = 0.0 if y_max == 0 else (val / y_max) * plot_h
        x = cx - bar_w / 2
        y = y0 - bh
        color = palette[i % len(palette)]

        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{y-8:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111">{val}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{y0+24:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#222">{escape(sched)}</text>'
        )

    # Axis labels
    parts.append(
        f'<text x="{left + plot_w/2:.1f}" y="{height-34}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111">Scheduler</text>'
    )
    parts.append(
        f'<text x="24" y="{top + plot_h/2:.1f}" transform="rotate(-90 24 {top + plot_h/2:.1f})" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#111">Unique Node Configurations</text>'
    )

    parts.append("</svg>")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    return target


def plot_counts(counts: dict[str, int], out_path: Path) -> Path:
    try:
        return _plot_counts_matplotlib(counts, out_path)
    except ModuleNotFoundError:
        return _plot_counts_svg(counts, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unique PP+DP node configuration counts per scheduler.",
    )
    parser.add_argument(
        "--db",
        default=str(BASELINES_DIR / "experiments-memorry.db"),
        help="Path to SQLite DB with planner_results.",
    )
    parser.add_argument(
        "--output",
        default=str(
            BASELINES_DIR
            / "experiment_data"
            / "reports"
            / "unique_node_config_counts.png"
        ),
        help="Output image path (.png).",
    )
    parser.add_argument(
        "--include-non-completed",
        action="store_true",
        help="Include runs whose status is not 'completed'.",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    out_path = Path(args.output).resolve()
    counts = load_unique_counts(
        db_path,
        completed_only=not args.include_non_completed,
    )
    actual_out = plot_counts(counts, out_path)

    print("Saved:", actual_out)
    for sched in sorted(counts):
        print(f"{sched}: {counts[sched]}")


if __name__ == "__main__":
    main()
