#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_name(name: str) -> tuple[int | None, int | None]:
    parts = name.split("/")
    if len(parts) >= 4 and parts[1] == "GemmPerf":
        partitions = int(parts[2])
        dim = int(parts[3])
        return partitions, dim
    if len(parts) >= 3 and parts[1] == "GemmScaling":
        partitions = int(parts[2])
        return partitions, None
    return None, None


def _real_time_ns(benchmark: Mapping[str, object]) -> float | None:
    real_time = _to_float(benchmark.get("real_time"))
    if real_time is None:
        return None

    unit = benchmark.get("time_unit")
    if unit == "ns":
        return real_time
    if unit == "us":
        return real_time * 1_000.0
    if unit == "ms":
        return real_time * 1_000_000.0
    if unit == "s":
        return real_time * 1_000_000_000.0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse bench_green_ctx GEMM JSON into CSV"
    )
    _ = parser.add_argument(
        "--input", required=True, help="Path to benchmark JSON"
    )
    _ = parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    input_path = cast(str, args.input)
    output_path = cast(str, args.output)

    raw_obj = cast(object, json.loads(Path(input_path).read_text()))
    if not isinstance(raw_obj, dict):
        raise ValueError("Invalid benchmark JSON: root is not an object")
    raw = cast(dict[str, object], raw_obj)

    benchmarks_obj = raw.get("benchmarks", [])
    if not isinstance(benchmarks_obj, list):
        raise ValueError("Invalid benchmark JSON: missing benchmarks array")
    benchmarks = cast(list[object], benchmarks_obj)

    rows: list[dict[str, object | None]] = []
    for bm_obj in benchmarks:
        if not isinstance(bm_obj, Mapping):
            continue
        bm = cast(Mapping[str, object], bm_obj)
        name_obj = bm.get("name")
        if not isinstance(name_obj, str):
            continue
        name = name_obj
        if not (
            name.startswith("GreenCtxGemmPerf/")
            or name.startswith("GreenCtxGemmScaling/")
        ):
            continue

        parsed_partitions, parsed_dim = _parse_name(name)
        partitions = _to_float(bm.get("Partitions"))
        if partitions is not None:
            num_partitions = int(partitions)
        else:
            num_partitions = parsed_partitions

        gflops = _to_float(bm.get("GFLOPS"))
        if gflops is None:
            gflops = _to_float(bm.get("Aggregate_GFLOPS"))

        rows.append(
            {
                "benchmark_name": name,
                "num_partitions": num_partitions,
                "dim": parsed_dim,
                "real_time_ns": _real_time_ns(bm),
                "GFLOPS": gflops,
                "SMs_assigned": _to_float(bm.get("SMs_assigned")),
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark_name",
                "num_partitions",
                "dim",
                "real_time_ns",
                "GFLOPS",
                "SMs_assigned",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
