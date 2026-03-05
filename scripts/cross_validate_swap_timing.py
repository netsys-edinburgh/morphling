#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import time
from typing import Protocol, TypedDict, cast


class SwapStatsProto(Protocol):
    count: int
    total_overhead_us: int

    def avg_overhead_us(self) -> float:
        ...


class RuntimeProto(Protocol):
    def activate_sm_for_thread(self, num_sms: int) -> int:
        ...

    def deactivate_for_thread(self, prev_sm_count: int) -> None:
        ...

    def swap_stats(self) -> SwapStatsProto:
        ...

    def close(self) -> None:
        ...


class InlineCppStats(TypedDict):
    total_swaps: int
    total_overhead_us: int
    avg_overhead_us: float


class InlinePythonStats(TypedDict):
    total_calls: int
    avg_us: float
    p50_us: float
    p99_us: float
    min_us: float
    max_us: float


class ComparisonStats(TypedDict):
    python_to_cpp_ratio: float
    within_5x: bool
    note: str


class Metadata(TypedDict):
    trace: str
    steps: int
    total_sms: int


class BaseResult(TypedDict):
    inline_cpp: InlineCppStats
    inline_python: InlinePythonStats


class ValidationResult(BaseResult, total=False):
    comparison: ComparisonStats
    metadata: Metadata


class ParsedArgs(Protocol):
    trace: str
    output: str
    steps: int
    total_sms: int


class GreenCtxModule(Protocol):
    def create_runtime(
        self,
        *,
        gpu_id: int,
        num_partitions: int,
        partition_idx: int,
    ) -> RuntimeProto:
        ...


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize_python(values_us: list[float]) -> InlinePythonStats:
    if not values_us:
        return {
            "total_calls": 0,
            "avg_us": 0.0,
            "p50_us": 0.0,
            "p99_us": 0.0,
            "min_us": 0.0,
            "max_us": 0.0,
        }
    total_calls = len(values_us)
    avg_us = float(sum(values_us) / total_calls)
    return {
        "total_calls": total_calls,
        "avg_us": avg_us,
        "p50_us": _percentile(values_us, 0.50),
        "p99_us": _percentile(values_us, 0.99),
        "min_us": float(min(values_us)),
        "max_us": float(max(values_us)),
    }


def _create_runtime() -> RuntimeProto:
    module = cast(
        GreenCtxModule,
        cast(object, importlib.import_module("morphling._GreenCtx")),
    )
    return module.create_runtime(gpu_id=0, num_partitions=1, partition_idx=0)


def _load_sm_counts(trace_path: str, max_rows: int) -> list[int]:
    sm_counts: list[int] = []
    with open(trace_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            raw_sm = row.get("sm_count")
            if raw_sm is None:
                continue
            sm_counts.append(int(raw_sm))
    return sm_counts


def measure_inline_timing(
    trace_path: str,
    steps: int,
    total_sms: int = 48,
) -> ValidationResult:
    runtime = _create_runtime()
    python_times_us: list[float] = []
    max_rows = max(0, steps * 100)
    sm_counts = _load_sm_counts(trace_path, max_rows)

    for sm in sm_counts:
        effective_sm = total_sms - sm
        if effective_sm <= 0:
            continue
        t0 = time.perf_counter_ns()
        prev = runtime.activate_sm_for_thread(effective_sm)
        t1 = time.perf_counter_ns()
        python_times_us.append((t1 - t0) / 1000.0)
        runtime.deactivate_for_thread(prev)

    stats = runtime.swap_stats()
    runtime.close()

    return {
        "inline_cpp": {
            "total_swaps": int(stats.count),
            "total_overhead_us": int(stats.total_overhead_us),
            "avg_overhead_us": float(stats.avg_overhead_us()),
        },
        "inline_python": _summarize_python(python_times_us),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--trace", default="data/ldpc_trace_with_ctrl.csv")
    _ = parser.add_argument(
        "--output",
        default="results/swap_timing_validation.json",
    )
    _ = parser.add_argument("--steps", type=int, default=5)
    _ = parser.add_argument("--total-sms", type=int, default=48)
    args = cast(ParsedArgs, cast(object, parser.parse_args()))

    trace = args.trace
    output = args.output
    steps = args.steps
    total_sms = args.total_sms

    print("=== Cross-validating swap timing ===")
    print(f"Trace: {trace}")
    print(f"Steps: {steps}")

    results = measure_inline_timing(trace, steps, total_sms)

    cpp_avg = results["inline_cpp"]["avg_overhead_us"]
    py_avg = results["inline_python"]["avg_us"]

    ratio = 0.0
    within_5x = False
    if cpp_avg > 0:
        ratio = py_avg / cpp_avg
        within_5x = ratio < 5.0

    results["comparison"] = {
        "python_to_cpp_ratio": ratio,
        "within_5x": within_5x,
        "note": "Python overhead includes pybind11 dispatch + GIL",
    }

    if cpp_avg > 0 and ratio >= 5.0:
        print(f"WARNING: Python/C++ ratio {ratio:.1f}x exceeds 5x threshold")

    results["metadata"] = {
        "trace": trace,
        "steps": steps,
        "total_sms": total_sms,
    }

    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
