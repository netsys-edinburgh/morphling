#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

COUNTERS = [
    "trace_entries",
    "violations",
    "violation_rate",
    "actual_ctx_switches",
    "switch_p50_ns",
    "switch_p99_ns",
    "switch_max_ns",
    "switch_min_ns",
    "budget_min_us",
    "budget_max_us",
    "gpu_total_SMs",
    "unique_sm_cfgs",
]


Scalar = int | float | str | bool


def _first_value(
    benchmarks: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> Scalar | None:
    for bm in benchmarks:
        for key in keys:
            if key in bm:
                value = bm[key]
                if isinstance(value, (int, float, str, bool)):
                    return value
    return None


def _summarize_trace(
    trace_type: str, json_path: str
) -> dict[str, Scalar | None]:
    raw_data_obj = cast(object, json.loads(Path(json_path).read_text()))
    if not isinstance(raw_data_obj, dict):
        raise ValueError(f"Invalid benchmark JSON at {json_path}")
    raw_data = cast(dict[str, object], raw_data_obj)

    raw_benchmarks_obj: object = raw_data.get("benchmarks", [])
    benchmarks: list[dict[str, object]] = []
    if isinstance(raw_benchmarks_obj, list):
        raw_benchmarks = cast(list[object], raw_benchmarks_obj)
        for item in raw_benchmarks:
            if isinstance(item, dict):
                benchmarks.append(cast(dict[str, object], item))

    replay: dict[str, object] | None = None
    for bm in benchmarks:
        name = bm.get("name", "")
        if isinstance(name, str) and "ReplayViolation" in name:
            replay = bm

    actual_ctx_switches = None
    if replay is not None:
        candidate = replay.get("actual_ctx_switches")
        if isinstance(candidate, (int, float, str, bool)):
            actual_ctx_switches = candidate

    row: dict[str, Scalar | None] = {
        "trace_type": trace_type,
        "benchmark_name": "ReplayViolation+ForcedSwitch",
        "trace_entries": _first_value(benchmarks, ["trace_entries"]),
        "violations": _first_value(benchmarks, ["violations"]),
        "violation_rate": _first_value(benchmarks, ["violation_rate"]),
        "actual_ctx_switches": actual_ctx_switches,
        "switch_p50_ns": _first_value(
            benchmarks, ["switch_p50_ns", "forced_switch_p50_ns"]
        ),
        "switch_p99_ns": _first_value(
            benchmarks, ["switch_p99_ns", "forced_switch_p99_ns"]
        ),
        "switch_max_ns": _first_value(
            benchmarks, ["switch_max_ns", "forced_switch_max_ns"]
        ),
        "switch_min_ns": _first_value(
            benchmarks, ["switch_min_ns", "forced_switch_min_ns"]
        ),
        "budget_min_us": _first_value(benchmarks, ["budget_min_us"]),
        "budget_max_us": _first_value(benchmarks, ["budget_max_us"]),
        "gpu_total_SMs": _first_value(benchmarks, ["gpu_total_SMs"]),
        "unique_sm_cfgs": _first_value(benchmarks, ["unique_sm_cfgs"]),
    }
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Parse bench_trace_switch JSON outputs into CSV"
    )
    _ = parser.add_argument(
        "--random", required=True, help="Path to random.json"
    )
    _ = parser.add_argument(
        "--shaped", required=True, help="Path to shaped.json"
    )
    _ = parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    random_path = cast(str, args.random)
    shaped_path = cast(str, args.shaped)
    output_path_arg = cast(str, args.output)

    rows = [
        _summarize_trace("random", random_path),
        _summarize_trace("shaped", shaped_path),
    ]

    fieldnames = ["trace_type", "benchmark_name", *COUNTERS]
    output_path = Path(output_path_arg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
