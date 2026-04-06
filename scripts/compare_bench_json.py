#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid benchmark JSON (top-level not object): {path}"
        )
    return data


def is_aggregate_name(name: str) -> bool:
    aggregate_suffixes = (
        "_mean",
        "_median",
        "_stddev",
        "_cv",
        "_min",
        "_max",
        "_p50",
        "_p90",
        "_p95",
        "_p99",
    )
    return name.endswith(aggregate_suffixes)


def build_name_map(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    benches = doc.get("benchmarks")
    if not isinstance(benches, list):
        raise ValueError("Missing or invalid 'benchmarks' array")

    out: Dict[str, Dict[str, Any]] = {}
    for row in benches:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name:
            continue
        if is_aggregate_name(name):
            continue
        if row.get("run_type") == "aggregate":
            continue
        out[name] = row
    return out


def extract_counter(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_time_us(row: Dict[str, Any]) -> float:
    if "real_time" not in row:
        raise ValueError(
            f"Missing real_time for benchmark: {row.get('name', '?')}"
        )
    real_time = row["real_time"]
    if not isinstance(real_time, (int, float)):
        raise ValueError(
            f"Invalid real_time type for benchmark: {row.get('name', '?')}"
        )
    unit = row.get("time_unit", "ns")
    if not isinstance(unit, str):
        unit = "ns"

    value = float(real_time)
    if unit == "us":
        return value
    if unit == "ns":
        return value / 1000.0
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    raise ValueError(
        f"Unsupported time_unit '{unit}' for benchmark {row.get('name', '?')}"
    )


def geometric_mean(values: List[float]) -> float:
    positives = [v for v in values if v > 0.0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def format_regression(name: str, speedup: float) -> str:
    slowdown = (1.0 / speedup) if speedup > 0.0 else float("inf")
    if math.isfinite(slowdown):
        return f"{name} ({slowdown:.3f}x slower)"
    return f"{name} (inf slower)"


def compare(
    mode_a_doc: Dict[str, Any], mode_b_doc: Dict[str, Any]
) -> Dict[str, Any]:
    a_map = build_name_map(mode_a_doc)
    b_map = build_name_map(mode_b_doc)

    common_names = sorted(set(a_map.keys()) & set(b_map.keys()))
    missing_in_b = sorted(set(a_map.keys()) - set(b_map.keys()))
    missing_in_a = sorted(set(b_map.keys()) - set(a_map.keys()))

    comparisons: List[Dict[str, Any]] = []
    speedups: List[float] = []
    regressions: List[str] = []

    best_benchmark = ""
    best_speedup = 0.0
    worst_benchmark = ""
    worst_speedup = float("inf")

    for name in common_names:
        a_row = a_map[name]
        b_row = b_map[name]
        a_us = extract_time_us(a_row)
        b_us = extract_time_us(b_row)
        speedup = (a_us / b_us) if b_us > 0.0 else 0.0

        a_gflops = extract_counter(a_row, "GFLOPS")
        b_gflops = extract_counter(b_row, "GFLOPS")

        gflops_improvement_pct = None
        if a_gflops is not None and b_gflops is not None and a_gflops != 0.0:
            gflops_improvement_pct = (
                (b_gflops - a_gflops) / abs(a_gflops) * 100.0
            )

        result = {
            "benchmark": name,
            "mode_a": {"real_time_us": a_us, "GFLOPS": a_gflops},
            "mode_b": {"real_time_us": b_us, "GFLOPS": b_gflops},
            "speedup": speedup,
            "gflops_improvement_pct": gflops_improvement_pct,
        }
        comparisons.append(result)
        speedups.append(speedup)

        if speedup > best_speedup:
            best_benchmark = name
            best_speedup = speedup
        if speedup < worst_speedup:
            worst_benchmark = name
            worst_speedup = speedup
        if speedup < 1.0:
            regressions.append(format_regression(name, speedup))

    summary = {
        "matched_benchmarks": len(common_names),
        "missing_in_mode_b": missing_in_b,
        "missing_in_mode_a": missing_in_a,
        "geometric_mean_speedup": geometric_mean(speedups),
        "best_case_speedup": best_speedup if comparisons else 0.0,
        "best_case_benchmark": best_benchmark if comparisons else "",
        "worst_case_speedup": worst_speedup if comparisons else 0.0,
        "worst_case_benchmark": worst_benchmark if comparisons else "",
        "regressions": regressions,
    }

    return {"comparison": comparisons, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Google Benchmark JSON outputs"
    )
    parser.add_argument("mode_a_json", type=Path)
    parser.add_argument("mode_b_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        if not args.mode_a_json.exists():
            raise FileNotFoundError(
                f"Mode A JSON not found: {args.mode_a_json}"
            )
        if not args.mode_b_json.exists():
            raise FileNotFoundError(
                f"Mode B JSON not found: {args.mode_b_json}"
            )

        a_doc = load_json(args.mode_a_json)
        b_doc = load_json(args.mode_b_json)
        output = compare(a_doc, b_doc)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"Wrote comparison: {args.output}")
        print(
            "Geo mean speedup: "
            f"{output['summary']['geometric_mean_speedup']:.4f}, "
            f"matched={output['summary']['matched_benchmarks']}"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
