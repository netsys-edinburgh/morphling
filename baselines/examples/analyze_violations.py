#!/usr/bin/env python3
"""Analyze SM partition violations from GEMM + step boundary logs.

A *violation* occurs when a GEMM operation spills past its
step boundary AND the combined SM count of adjacent steps
exceeds a configurable threshold (default 48).

Inputs
------
--gemm-log   : CSV from the LD_PRELOAD GEMM interceptor
--step-log   : CSV of step boundaries emitted by the trainer
--trace      : (optional) original green-context trace CSV
--max-sm-count : SM threshold (default 48)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


# ── dataclasses ──────────────────────────────────────────


@dataclass
class GemmCall:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    m: int
    n: int
    k: int


@dataclass
class StepSlot:
    step: int
    start_ns: int
    end_ns: int
    sm_count: int


@dataclass
class Violation:
    step: int
    next_step: int
    sm_this: int
    sm_next: int
    gemm: GemmCall
    overshoot_ns: int  # how far past boundary


@dataclass
class StepViolationSummary:
    step: int
    next_step: int
    sm_this: int
    sm_next: int
    sm_total: int
    num_violations: int = 0
    max_overshoot_ns: int = 0
    total_overshoot_ns: int = 0
    violations: list[Violation] = field(
        default_factory=list
    )


# ── parsing ──────────────────────────────────────────────


def parse_gemm_log(path: Path) -> list[GemmCall]:
    """Parse GEMM timing CSV."""
    calls: list[GemmCall] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                calls.append(
                    GemmCall(
                        name=row["function_name"],
                        start_ns=int(row["start_ns"]),
                        end_ns=int(row["end_ns"]),
                        duration_ns=int(
                            row["duration_ns"]
                        ),
                        m=int(row.get("m", 0)),
                        n=int(row.get("n", 0)),
                        k=int(row.get("k", 0)),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.debug("skip malformed row: %s", exc)
    calls.sort(key=lambda c: c.start_ns)
    return calls


def parse_step_boundaries(
    path: Path,
) -> list[StepSlot]:
    """Parse step boundary CSV."""
    slots: list[StepSlot] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slots.append(
                StepSlot(
                    step=int(row["step"]),
                    start_ns=int(row["start_ns"]),
                    end_ns=int(row["end_ns"]),
                    sm_count=int(row["sm_count"]),
                )
            )
    slots.sort(key=lambda s: s.start_ns)
    return slots


def parse_autograd_log(path: Path) -> list[GemmCall]:
    """Parse autograd GEMM log (JSON from morphling hook)."""
    with path.open(encoding="utf-8") as f:
        entries = json.load(f)
    calls: list[GemmCall] = []
    for e in entries:
        start_ns = int(e["start_us"] * 1000)
        end_ns = int(e["end_us"] * 1000)
        calls.append(
            GemmCall(
                name=e.get("phase", "unknown"),
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                m=int(e.get("m", 0)),
                n=int(e.get("n", 0)),
                k=int(e.get("k", 0)),
            )
        )
    calls.sort(key=lambda c: c.start_ns)
    return calls


def parse_trace_boundaries(path: Path) -> list[StepSlot]:
    """Parse LDPC trace CSV as slot boundaries with SM counts."""
    import pandas as pd

    df: Any = pd.read_csv(path)
    slots: list[StepSlot] = []
    for i, row in df.iterrows():
        slots.append(
            StepSlot(
                step=int(i),
                start_ns=int(row["time_slot_sched_ns"]),
                end_ns=0,
                sm_count=int(row["sm_count"]),
            )
        )
    for i in range(len(slots) - 1):
        slots[i] = StepSlot(
            step=slots[i].step,
            start_ns=slots[i].start_ns,
            end_ns=slots[i + 1].start_ns,
            sm_count=slots[i].sm_count,
        )
    if slots:
        last = slots[-1]
        slots[-1] = StepSlot(
            step=last.step,
            start_ns=last.start_ns,
            end_ns=last.start_ns + 1_000_000,
            sm_count=last.sm_count,
        )
    return slots


def parse_trace(path: Path) -> list[tuple[int, int]]:
    """Parse green-context trace CSV (optional)."""
    entries: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                ts = int(parts[0].strip())
                sms = int(parts[1].strip())
                entries.append((ts, sms))
            except ValueError:
                continue
    return entries


# ── assignment ───────────────────────────────────────────


def assign_gemms_to_steps(
    gemms: list[GemmCall],
    slots: list[StepSlot],
) -> dict[int, list[GemmCall]]:
    """Map each GEMM to a step via binary search."""
    starts = [s.start_ns for s in slots]
    result: dict[int, list[GemmCall]] = {
        i: [] for i in range(len(slots))
    }
    for g in gemms:
        idx = bisect.bisect_right(starts, g.start_ns) - 1
        if 0 <= idx < len(slots):
            result[idx].append(g)
    return result


# ── violation detection ──────────────────────────────────


def detect_violations(
    slots: list[StepSlot],
    step_gemms: dict[int, list[GemmCall]],
    max_sm: int,
) -> list[StepViolationSummary]:
    """Find violations at step boundaries."""
    summaries: list[StepViolationSummary] = []

    for i in range(len(slots) - 1):
        cur = slots[i]
        nxt = slots[i + 1]
        sm_total = cur.sm_count + nxt.sm_count

        summary = StepViolationSummary(
            step=cur.step,
            next_step=nxt.step,
            sm_this=cur.sm_count,
            sm_next=nxt.sm_count,
            sm_total=sm_total,
        )

        if sm_total <= max_sm:
            continue

        for g in step_gemms.get(i, []):
            if g.end_ns > nxt.start_ns:
                overshoot = g.end_ns - nxt.start_ns
                v = Violation(
                    step=cur.step,
                    next_step=nxt.step,
                    sm_this=cur.sm_count,
                    sm_next=nxt.sm_count,
                    gemm=g,
                    overshoot_ns=overshoot,
                )
                summary.num_violations += 1
                summary.total_overshoot_ns += overshoot
                summary.max_overshoot_ns = max(
                    summary.max_overshoot_ns, overshoot
                )
                summary.violations.append(v)

        if summary.num_violations > 0:
            summaries.append(summary)

    return summaries


# ── violation time (sweepline) ───────────────────────────


def compute_violation_time_ns(
    summaries: list[StepViolationSummary],
    slots: list[StepSlot],
) -> int:
    """Merge overlapping violation intervals."""
    intervals: list[tuple[int, int]] = []
    slot_map = {s.step: s for s in slots}

    for s in summaries:
        nxt = slot_map.get(s.next_step)
        if nxt is None:
            continue
        for v in s.violations:
            start = nxt.start_ns
            end = min(v.gemm.end_ns, nxt.end_ns)
            if end > start:
                intervals.append((start, end))

    if not intervals:
        return 0

    intervals.sort()
    merged: list[tuple[int, int]] = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], e),
            )
        else:
            merged.append((s, e))

    return sum(e - s for s, e in merged)


# ── reporting ────────────────────────────────────────────


def format_ns(ns: int) -> str:
    """Human-readable nanosecond duration."""
    if ns < 1_000:
        return f"{ns}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f}us"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    return f"{ns / 1_000_000_000:.3f}s"


def write_report(
    out: TextIO,
    gemms: list[GemmCall],
    slots: list[StepSlot],
    summaries: list[StepViolationSummary],
    violation_time_ns: int,
    max_sm: int,
    top_n: int,
    verbose: bool,
) -> None:
    """Write text report."""
    total_steps = len(slots)
    total_boundaries = max(total_steps - 1, 1)
    total_wall_ns = 0
    if slots:
        total_wall_ns = (
            slots[-1].end_ns - slots[0].start_ns
        )

    total_violating_gemms = sum(
        s.num_violations for s in summaries
    )
    slots_with_v = len(summaries)

    pct_slots = (
        100.0 * slots_with_v / total_boundaries
        if total_boundaries
        else 0.0
    )
    pct_time = (
        100.0 * violation_time_ns / total_wall_ns
        if total_wall_ns
        else 0.0
    )

    out.write(
        "=== SM Partition Violation Analysis ===\n\n"
    )
    out.write("Configuration:\n")
    out.write(f"  Max SM count threshold: {max_sm}\n")
    out.write(f"  GEMM log entries: {len(gemms)}\n")
    out.write(f"  Training steps: {total_steps}\n")
    out.write(
        f"  Total wall time: "
        f"{format_ns(total_wall_ns)}\n\n"
    )
    out.write("Summary:\n")
    out.write(
        f"  Slots with violations: "
        f"{slots_with_v}/{total_boundaries} "
        f"({pct_slots:.2f}%)\n"
    )
    out.write(
        f"  Time with violations: "
        f"{format_ns(violation_time_ns)} / "
        f"{format_ns(total_wall_ns)} "
        f"({pct_time:.4f}%)\n"
    )
    out.write(
        f"  Total violating GEMM calls: "
        f"{total_violating_gemms}/{len(gemms)} "
        f"({100.0 * total_violating_gemms / max(len(gemms), 1):.2f}%)\n"
    )
    out.write("\n")

    if not summaries:
        out.write("No violations detected.\n")
        return

    ranked = sorted(
        summaries,
        key=lambda s: s.max_overshoot_ns,
        reverse=True,
    )

    show = ranked if verbose else ranked[:top_n]
    out.write(
        f"Per-step violations "
        f"(top {len(show)} by overshoot):\n"
    )
    for s in show:
        out.write(
            f"  Step {s.step}->{s.next_step}: "
            f"{s.num_violations} violations, "
            f"max_overshoot="
            f"{format_ns(s.max_overshoot_ns)}, "
            f"sm_pressure="
            f"{s.sm_this}+{s.sm_next}="
            f"{s.sm_total} > {max_sm}\n"
        )
    out.write("\n")

    limit = None if verbose else 20
    detail_count = 0
    out.write(
        f"Detailed violations "
        f"(first {limit or 'all'}):\n"
    )
    for s in ranked:
        for v in s.violations:
            out.write(
                f"  step={v.step} "
                f"gemm={v.gemm.name} "
                f"m={v.gemm.m} n={v.gemm.n} "
                f"k={v.gemm.k} "
                f"overshoot="
                f"{format_ns(v.overshoot_ns)} "
                f"sm={v.sm_this}+{v.sm_next}="
                f"{v.sm_this + v.sm_next}\n"
            )
            detail_count += 1
            if limit and detail_count >= limit:
                break
        if limit and detail_count >= limit:
            break
    out.write("\n")


def write_csv_summary(
    path: Path,
    summaries: list[StepViolationSummary],
) -> None:
    """Write per-step violation CSV."""
    fieldnames = [
        "step",
        "next_step",
        "sm_this",
        "sm_next",
        "sm_total",
        "num_violations",
        "max_overshoot_ns",
        "total_overshoot_ns",
    ]
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "step": s.step,
                    "next_step": s.next_step,
                    "sm_this": s.sm_this,
                    "sm_next": s.sm_next,
                    "sm_total": s.sm_total,
                    "num_violations": s.num_violations,
                    "max_overshoot_ns": (
                        s.max_overshoot_ns
                    ),
                    "total_overshoot_ns": (
                        s.total_overshoot_ns
                    ),
                }
            )


# ── CLI ──────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SM partition violation analysis",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="ldpreload",
        choices=["ldpreload", "autograd"],
        help="Input mode: ldpreload CSVs or autograd JSON+trace",
    )
    p.add_argument(
        "--gemm-log",
        type=str,
        default=None,
        help="Path to GEMM timing CSV",
    )
    p.add_argument(
        "--autograd-log",
        type=str,
        default=None,
        help="Path to autograd GEMM JSON log",
    )
    p.add_argument(
        "--step-log",
        type=str,
        default=None,
        help="Path to step boundaries CSV",
    )
    p.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Original green-context trace CSV",
    )
    p.add_argument(
        "--max-sm-count",
        type=int,
        default=48,
        help="SM threshold for violation",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Text report output (default stdout)",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="CSV summary output path",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="JSON summary output path",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top violations to show",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show all violations",
    )
    return p.parse_args()


def write_json_summary(
    path: Path,
    gemms: list[GemmCall],
    slots: list[StepSlot],
    summaries: list[StepViolationSummary],
    violation_time_ns: int,
) -> None:
    total_boundaries = max(len(slots) - 1, 1)
    result = {
        "total_slots": len(slots),
        "slots_with_violations": len(summaries),
        "violation_pct": 100.0
        * len(summaries)
        / total_boundaries,
        "total_gemms": len(gemms),
        "violating_gemms": sum(
            s.num_violations for s in summaries
        ),
        "total_violation_time_ns": violation_time_ns,
        "per_step_violations": [
            {
                "step": s.step,
                "violations": s.num_violations,
                "max_overshoot_ns": s.max_overshoot_ns,
                "total_overshoot_ns": s.total_overshoot_ns,
            }
            for s in summaries
        ],
    }
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(name)s "
            "%(levelname)s %(message)s"
        ),
    )
    args = parse_args()

    if args.mode == "autograd":
        if not args.autograd_log:
            logger.error(
                "--autograd-log is required in autograd mode"
            )
            sys.exit(1)
        if not args.trace:
            logger.error("--trace is required in autograd mode")
            sys.exit(1)

        autograd_path = Path(args.autograd_log)
        trace_path = Path(args.trace)

        if not autograd_path.exists():
            logger.error(
                "Autograd log not found: %s", autograd_path
            )
            sys.exit(1)
        if not trace_path.exists():
            logger.error("Trace not found: %s", trace_path)
            sys.exit(1)

        logger.info("parsing autograd log: %s", autograd_path)
        gemms = parse_autograd_log(autograd_path)
        logger.info("  %d GEMM calls loaded", len(gemms))

        logger.info(
            "parsing trace boundaries: %s", trace_path
        )
        slots = parse_trace_boundaries(trace_path)
        logger.info("  %d steps loaded", len(slots))
    else:
        if not args.gemm_log:
            logger.error("--gemm-log is required in ldpreload mode")
            sys.exit(1)
        if not args.step_log:
            logger.error("--step-log is required in ldpreload mode")
            sys.exit(1)

        gemm_path = Path(args.gemm_log)
        step_path = Path(args.step_log)

        if not gemm_path.exists():
            logger.error("GEMM log not found: %s", gemm_path)
            sys.exit(1)
        if not step_path.exists():
            logger.error(
                "Step boundaries not found: %s", step_path
            )
            sys.exit(1)

        logger.info("parsing GEMM log: %s", gemm_path)
        gemms = parse_gemm_log(gemm_path)
        logger.info("  %d GEMM calls loaded", len(gemms))

        logger.info(
            "parsing step boundaries: %s", step_path
        )
        slots = parse_step_boundaries(step_path)
        logger.info("  %d steps loaded", len(slots))

        if args.trace:
            trace_path = Path(args.trace)
            if trace_path.exists():
                trace = parse_trace(trace_path)
                logger.info(
                    "trace loaded: %d entries", len(trace)
                )

    if len(slots) < 2:
        logger.warning(
            "Need >= 2 steps for violation analysis"
        )
        print("No violations (insufficient steps).")
        return

    logger.info("assigning GEMMs to steps...")
    step_gemms = assign_gemms_to_steps(gemms, slots)

    logger.info(
        "detecting violations "
        "(max_sm=%d)...",
        args.max_sm_count,
    )
    summaries = detect_violations(
        slots, step_gemms, args.max_sm_count
    )

    violation_time = compute_violation_time_ns(
        summaries, slots
    )

    out: TextIO
    if args.output:
        out = open(
            args.output, "w", encoding="utf-8"
        )
    else:
        out = sys.stdout

    try:
        write_report(
            out=out,
            gemms=gemms,
            slots=slots,
            summaries=summaries,
            violation_time_ns=violation_time,
            max_sm=args.max_sm_count,
            top_n=args.top_n,
            verbose=args.verbose,
        )
    finally:
        if args.output:
            out.close()
            logger.info(
                "wrote report: %s", args.output
            )

    if args.output_csv:
        csv_path = Path(args.output_csv)
        write_csv_summary(csv_path, summaries)
        logger.info("wrote CSV summary: %s", csv_path)

    if args.output_json:
        json_path = Path(args.output_json)
        write_json_summary(
            json_path,
            gemms,
            slots,
            summaries,
            violation_time,
        )
        logger.info("wrote JSON summary: %s", json_path)


if __name__ == "__main__":
    main()
