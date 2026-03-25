#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

Numeric = Union[int, float]


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((x - mu) ** 2 for x in values) / len(values)))


def parse_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def is_warmup_row(row: Dict[str, str]) -> bool:
    raw = row.get("is_warmup")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "t", "yes", "y"}


def pick_metrics_csv(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "eval_metrics_greenctx.csv",
        run_dir / "metrics.csv",
        run_dir / "eval_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    print(f"  WARNING: No metrics CSV in {run_dir}")
    return None


def load_run_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    csv_path = pick_metrics_csv(run_dir)
    if csv_path is None:
        return None

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row and not is_warmup_row(row)]

    if not rows:
        print(f"  WARNING: Empty metrics rows in {csv_path}")
        return None

    tokens_per_sec = [parse_float(row, "tokens_per_sec") for row in rows]
    wall_time_ms = [parse_float(row, "wall_time_ms") for row in rows]

    result: Dict[str, float] = {
        "tokens_per_sec_mean": mean(tokens_per_sec),
        "step_time_ms_mean": mean(wall_time_ms),
        "step_time_ms_min": float(min(wall_time_ms)) if wall_time_ms else 0.0,
        "step_time_ms_max": float(max(wall_time_ms)) if wall_time_ms else 0.0,
    }

    if any("swap_count" in row for row in rows):
        swap_counts = [parse_float(row, "swap_count") for row in rows]
        result["swap_count_mean"] = mean(swap_counts)

    if any("swap_overhead_us" in row for row in rows):
        swap_overheads = [parse_float(row, "swap_overhead_us") for row in rows]
        result["swap_overhead_us_mean"] = mean(swap_overheads)

    if any("gemm_count" in row for row in rows):
        gemm_counts = [parse_float(row, "gemm_count") for row in rows]
        result["gemm_count_mean"] = mean(gemm_counts)

    return result


def load_violation_results(
    run_dir: Path,
) -> Optional[Dict[str, object]]:
    for name in ["violations.json", "violation_summary.json"]:
        path = run_dir / name
        if path.exists():
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    return None


def aggregate_config(config_dir: Path) -> Dict[str, object]:
    run_dirs = sorted(
        [path for path in config_dir.glob("run_*") if path.is_dir()]
    )
    if not run_dirs:
        print(f"  No runs found in {config_dir}")
        return {}

    metrics_list: List[Dict[str, float]] = []
    violation_list: List[Dict[str, object]] = []

    for run_dir in run_dirs:
        metrics = load_run_metrics(run_dir)
        if metrics is not None:
            metrics_list.append(metrics)

        violation = load_violation_results(run_dir)
        if violation is not None:
            violation_list.append(violation)

    if not metrics_list:
        return {}

    aggregate: Dict[str, object] = {}
    all_keys = sorted({key for item in metrics_list for key in item.keys()})
    for key in all_keys:
        values = [item[key] for item in metrics_list if key in item]
        if values:
            aggregate[key] = mean(values)
            aggregate[f"{key}_std"] = std(values)

    step_times = [item.get("step_time_ms_mean", 0.0) for item in metrics_list]
    aggregate["step_time_ms_run_min"] = float(min(step_times))
    aggregate["step_time_ms_run_max"] = float(max(step_times))

    if violation_list:
        violations_per_step: List[float] = []
        violation_times_ns: List[float] = []
        violation_counts: List[float] = []
        avg_violation_duration_us: List[float] = []
        for violation in violation_list:
            total_violations = numeric(
                violation.get(
                    "total_violations",
                    violation.get("violating_gemms", 0),
                )
            )
            total_slots = numeric(violation.get("total_slots", 1))
            violations_per_step.append(total_violations / max(total_slots, 1.0))
            violation_times_ns.append(
                numeric(violation.get("total_violation_time_ns", 0))
            )
            violation_counts.append(total_violations)
            avg_violation_duration_us.append(
                numeric(
                    violation.get(
                        "avg_violation_duration_us",
                        0,
                    )
                )
            )

        aggregate["violations_per_step_mean"] = mean(violations_per_step)
        aggregate["total_violation_time_ns_mean"] = mean(violation_times_ns)
        aggregate["total_violations_mean"] = mean(violation_counts)
        aggregate["avg_violation_duration_us_mean"] = mean(
            avg_violation_duration_us
        )

    aggregate["n_runs"] = len(metrics_list)
    return aggregate


def numeric(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def add_overhead_decomposition(
    paper_data: Dict[str, Dict[str, object]],
) -> None:
    for config in ["without_ctrl", "with_ctrl"]:
        raw = paper_data.get(config)
        if not isinstance(raw, dict):
            continue

        swap_overhead_us_mean = numeric(raw.get("swap_overhead_us_mean", 0))
        swap_overhead_ms = swap_overhead_us_mean / 1000.0
        swap_count = numeric(raw.get("swap_count_mean", 0))
        violation_time_ms = (
            numeric(raw.get("total_violation_time_ns_mean", 0)) / 1e6
        )

        raw["swap_overhead_ms_per_step"] = swap_overhead_ms
        raw["violation_time_ms_per_step"] = violation_time_ms
        raw["combined_overhead_ms"] = swap_overhead_ms + violation_time_ms
        raw["avg_swap_overhead_us"] = swap_overhead_us_mean / max(
            swap_count, 1.0
        )


def _nearest_key(value: float, keys: List[int]) -> int:
    if not keys:
        return 0
    return min(keys, key=lambda k: abs(k - value))


def _load_sm_counts_from_metrics(csv_path: Path) -> List[float]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row and not is_warmup_row(row)]
    return [parse_float(row, "sm_count") for row in rows if "sm_count" in row]


def add_gflops_estimates(
    paper_data: Dict[str, Dict[str, object]],
    results_dir: Path,
    gflops_per_step: Dict[str, object],
) -> None:
    gflops_lookup: Dict[int, float] = {}
    for key, payload in gflops_per_step.items():
        try:
            sm_key = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            gflops_lookup[sm_key] = numeric(payload.get("total_gflops", 0))

    benchmark_keys = sorted(gflops_lookup.keys())
    for config in ["without_ctrl", "with_ctrl"]:
        config_dir = results_dir / config
        gflops_samples: List[float] = []
        for run_dir in sorted(
            [path for path in config_dir.glob("run_*") if path.is_dir()]
        ):
            csv_path = pick_metrics_csv(run_dir)
            if csv_path is None:
                continue
            for sm_count in _load_sm_counts_from_metrics(csv_path):
                nearest = _nearest_key(sm_count, benchmark_keys)
                if nearest in gflops_lookup:
                    gflops_samples.append(gflops_lookup[nearest])

        if config not in paper_data or not isinstance(paper_data[config], dict):
            paper_data[config] = {}
        payload = paper_data[config]
        payload["gflops_mean"] = mean(gflops_samples)
        payload["gflops_mean_std"] = std(gflops_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--results-dir", default="results")
    _ = parser.add_argument(
        "--gflops",
        default="results/gflops_per_sm.json",
        help="GFLOPS benchmark data",
    )
    _ = parser.add_argument("--output", default="results/paper_data.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    paper_data: Dict[str, Dict[str, object]] = {}

    for config in ["without_ctrl", "with_ctrl"]:
        config_dir = results_dir / config
        print(f"\n=== Aggregating {config} ===")
        paper_data[config] = aggregate_config(config_dir)

    if os.path.exists(args.gflops):
        with open(args.gflops, encoding="utf-8") as f:
            gflops_data = json.load(f)
        if isinstance(gflops_data, dict):
            per_step = gflops_data.get("per_step_aggregate", {})
            paper_data["gflops_benchmark"] = per_step
            if isinstance(per_step, dict):
                add_gflops_estimates(paper_data, results_dir, per_step)

    add_overhead_decomposition(paper_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(paper_data, f, indent=2)
    print(f"\nPaper data saved to {output_path}")

    for config in ["without_ctrl", "with_ctrl"]:
        data = paper_data.get(config, {})
        if not isinstance(data, dict):
            data = {}

        print(f"\n{config}:")
        print(f"  Runs: {data.get('n_runs', 0)}")
        print(
            f"  Throughput: {numeric(data.get('tokens_per_sec_mean', 0)):.1f} "
            "tok/s"
        )
        print(
            f"  Step time: {numeric(data.get('step_time_ms_mean', 0)):.1f} ms"
        )
        print(
            f"  Swap count: {numeric(data.get('swap_count_mean', 0)):.1f}/step"
        )
        print(
            "  Combined overhead: "
            f"{numeric(data.get('combined_overhead_ms', 0)):.3f} ms"
        )


if __name__ == "__main__":
    main()
