from __future__ import annotations

import csv
import json
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Any, cast

from .bootstrap import REPO_ROOT

WARMUP_STEPS = 5


def _write_violation_artifacts(
    *,
    run_dir: Path,
    metrics_df: Any,
    gemm_log: list[dict[str, Any]],
    max_sm_count: int,
) -> None:
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "analyze_violations",
        REPO_ROOT / "scripts" / "analyze_violations.py",
    )
    _mod = _ilu.module_from_spec(cast(ModuleSpec, _spec))
    sys.modules.setdefault("analyze_violations", _mod)
    cast(Any, cast(ModuleSpec, _spec).loader).exec_module(_mod)
    av = _mod

    required_cols = {"step_idx", "step_start_ns", "step_end_ns", "sm_count"}
    if not required_cols.issubset(set(metrics_df.columns)):
        raise ValueError(
            "Missing required columns for violation artifacts: "
            + ", ".join(sorted(required_cols - set(metrics_df.columns)))
        )

    step_path = run_dir / "step_boundaries.csv"
    step_df = metrics_df[["step_idx", "step_start_ns", "step_end_ns", "sm_count"]].copy()
    step_df.columns = ["step", "start_ns", "end_ns", "sm_count"]
    step_df.to_csv(step_path, index=False)

    gemm_path = run_dir / "gemm_log.csv"
    epoch_ns = int(step_df["start_ns"].iloc[0]) if len(step_df) > 0 else 0
    with open(gemm_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["function_name", "start_ns", "end_ns", "duration_ns", "m", "n", "k"],
        )
        writer.writeheader()
        for entry in gemm_log:
            start_ns = epoch_ns + int(float(entry.get("start_us", 0.0)) * 1000.0)
            end_ns = epoch_ns + int(float(entry.get("end_us", 0.0)) * 1000.0)
            if end_ns < start_ns:
                end_ns = start_ns
            writer.writerow(
                {
                    "function_name": str(entry.get("phase", "gemm")),
                    "start_ns": int(start_ns),
                    "end_ns": int(end_ns),
                    "duration_ns": int(end_ns - start_ns),
                    "m": int(entry.get("m", 0)),
                    "n": int(entry.get("n", 0)),
                    "k": int(entry.get("k", 0)),
                }
            )

    gemms = av.parse_gemm_log(gemm_path)
    slots = av.parse_step_boundaries(step_path)
    step_gemms = av.assign_gemms_to_steps(gemms, slots)
    summaries = av.detect_violations(slots, step_gemms, max_sm_count)
    violation_time_ns = av.compute_violation_time_ns(summaries, slots)

    report_path = run_dir / "violation_report.txt"
    with open(report_path, "w", encoding="utf-8") as out:
        av.write_report(
            out=out,
            gemms=gemms,
            slots=slots,
            summaries=summaries,
            violation_time_ns=violation_time_ns,
            max_sm=max_sm_count,
            top_n=10,
            verbose=False,
        )

    av.write_csv_summary(run_dir / "violation_summary.csv", summaries)
    av.write_json_summary(
        run_dir / "violations.json",
        gemms,
        slots,
        summaries,
        violation_time_ns,
    )


def _timed_rows(df: Any) -> Any:
    timed = df.loc[df["step_idx"] >= WARMUP_STEPS]
    if len(timed) == 0:
        return df
    return timed


def _summarize_run(df: Any) -> dict[str, float]:
    timed = _timed_rows(df)
    return {
        "tokens_per_sec": float(timed["tokens_per_sec"].mean()),
        "step_time_ms": float(timed["wall_time_ms"].mean()),
        "swap_count": float(timed["swap_count"].mean()),
        "swap_overhead_us": float(timed["swap_overhead_us"].mean()),
    }


def _dump_gemm_shapes(
    output_dir: str,
    num_steps: int,
    gemm_log: list[dict[str, Any]] | None = None,
) -> None:
    import os
    from collections import Counter

    if gemm_log is None:
        from morphling.hooks.autograd import get_gemm_log

        log = get_gemm_log()
    else:
        log = gemm_log
    shape_counts = Counter()
    for entry in log:
        key = (entry["m"], entry["n"], entry["k"], entry["phase"])
        shape_counts[key] += 1

    shapes = []
    for (m, n, k, phase), count in shape_counts.most_common():
        shapes.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "phase": phase,
                "count_per_step": count / max(num_steps, 1),
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "gemm_shapes.json")
    with open(path, "w") as f:
        json.dump(shapes, f, indent=2)
    print(f"Saved {len(shapes)} GEMM shapes to {path}")
