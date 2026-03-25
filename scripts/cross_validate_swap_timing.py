#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Mapping
from typing import Protocol, cast


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


class GreenCtxModule(Protocol):
    def create_runtime(
        self,
        *,
        gpu_id: int,
        num_partitions: int,
        partition_idx: int,
    ) -> RuntimeProto:
        ...


class ParsedArgs(Protocol):
    trace: str | None
    trace_with: str | None
    trace_without: str | None
    output: str
    steps: int
    total_sms: int
    benchmark_bin: str | None
    bench_json_with: str | None
    bench_json_without: str | None
    benchmark_timeout_s: int


class TraceArgsProto(Protocol):
    trace: str | None
    trace_with: str | None
    trace_without: str | None


def _bootstrap_morphling_runtime(repo_root: Path) -> None:
    package_paths: list[str] = [str(repo_root / "morphling")]
    for entry in sys.path:
        candidate = Path(entry) / "morphling"
        if candidate.is_dir():
            path_str = str(candidate)
            if path_str not in package_paths:
                package_paths.append(path_str)

    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = package_paths
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod
    else:
        morphling_mod = sys.modules["morphling"]
        current_paths = list(getattr(morphling_mod, "__path__", []))
        for path_str in package_paths:
            if path_str not in current_paths:
                current_paths.append(path_str)
        morphling_mod.__path__ = current_paths

    if "morphling._GreenCtx" in sys.modules:
        return

    so_candidates: list[Path] = []
    for pkg_dir in package_paths:
        so_candidates.extend(sorted(Path(pkg_dir).glob("_GreenCtx*.so")))

    for so_path in so_candidates:
        try:
            spec = importlib.util.spec_from_file_location(
                "morphling._GreenCtx", str(so_path)
            )
            if spec is None or spec.loader is None:
                continue
            green_ctx_mod = importlib.util.module_from_spec(spec)
            sys.modules["morphling._GreenCtx"] = green_ctx_mod
            spec.loader.exec_module(green_ctx_mod)
            return
        except Exception:
            _ = sys.modules.pop("morphling._GreenCtx", None)
            continue


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


def _summarize_overhead_us(values_us: list[float]) -> dict[str, float | int]:
    if not values_us:
        return {
            "count": 0,
            "total_overhead_us": 0.0,
            "avg_overhead_us": 0.0,
            "p50_overhead_us": 0.0,
            "p99_overhead_us": 0.0,
            "min_overhead_us": 0.0,
            "max_overhead_us": 0.0,
        }
    total = float(sum(values_us))
    count = len(values_us)
    return {
        "count": count,
        "total_overhead_us": total,
        "avg_overhead_us": float(total / count),
        "p50_overhead_us": _percentile(values_us, 0.50),
        "p99_overhead_us": _percentile(values_us, 0.99),
        "min_overhead_us": float(min(values_us)),
        "max_overhead_us": float(max(values_us)),
    }


def _create_runtime() -> RuntimeProto:
    repo_root = Path(__file__).resolve().parents[1]
    _bootstrap_morphling_runtime(repo_root)
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


def _as_float(raw: object, default: float = 0.0) -> float:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            return float(raw)
        except ValueError:
            return default
    return default


def _find_first_numeric(
    row: Mapping[str, object],
    keys: tuple[str, ...],
) -> float:
    for key in keys:
        if key in row:
            return _as_float(row.get(key))
    return 0.0


def _parse_replay_violation_bench_json(
    bench_json_path: str,
) -> dict[str, float | int | str]:
    with open(bench_json_path, encoding="utf-8") as f:
        raw_obj = cast(object, json.load(f))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"Invalid benchmark JSON: {bench_json_path}")
    raw = cast(dict[str, object], raw_obj)

    benches_obj = raw.get("benchmarks", [])
    if not isinstance(benches_obj, list):
        raise ValueError(f"Invalid benchmark JSON: {bench_json_path}")

    benches: list[dict[str, object]] = []
    raw_benches = cast(list[object], benches_obj)
    for item_obj in raw_benches:
        if isinstance(item_obj, dict):
            benches.append(cast(dict[str, object], item_obj))

    replay: dict[str, object] | None = None
    for row in benches:
        name = row.get("name")
        if isinstance(name, str) and "ReplayViolation" in name:
            lowered = name.lower()
            if (
                "_mean" in lowered
                or "_median" in lowered
                or "_stddev" in lowered
            ):
                continue
            replay = row
            break
    if replay is None:
        for row in benches:
            name = row.get("name")
            if isinstance(name, str) and "ReplayViolation" in name:
                replay = row
                break
    if replay is None:
        raise ValueError(
            f"ReplayViolation benchmark row not found in {bench_json_path}"
        )

    p50_ns = _find_first_numeric(
        replay,
        ("switch_p50_ns", "forced_switch_p50_ns"),
    )
    p99_ns = _find_first_numeric(
        replay,
        ("switch_p99_ns", "forced_switch_p99_ns"),
    )
    min_ns = _find_first_numeric(
        replay,
        ("switch_min_ns", "forced_switch_min_ns"),
    )
    max_ns = _find_first_numeric(
        replay,
        ("switch_max_ns", "forced_switch_max_ns"),
    )

    return {
        "trace_entries": int(_find_first_numeric(replay, ("trace_entries",))),
        "violations": int(_find_first_numeric(replay, ("violations",))),
        "violation_rate": float(_find_first_numeric(replay, ("violation_rate",))),
        "actual_ctx_switches": int(
            _find_first_numeric(replay, ("actual_ctx_switches",))
        ),
        "switch_p50_ns": float(p50_ns),
        "switch_p99_ns": float(p99_ns),
        "switch_min_ns": float(min_ns),
        "switch_max_ns": float(max_ns),
        "switch_p50_us": float(p50_ns / 1e3),
        "switch_p99_us": float(p99_ns / 1e3),
    }


def _resolve_benchmark_bin(explicit_path: str | None) -> str | None:
    candidates: list[str] = []
    if explicit_path:
        candidates.append(explicit_path)
    env_bin = os.environ.get("BENCH_TRACE_SWITCH_BIN")
    if env_bin:
        candidates.append(env_bin)
    candidates.extend(
        [
            "tests/cpp/build/bench_trace_switch",
            "/app/tests/cpp/build/bench_trace_switch",
        ]
    )
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _discover_bench_json_path(
    label: str,
    explicit_path: str | None,
) -> str | None:
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    candidates = [
        f"results/bench_trace_switch_{label}.json",
        f"results/trace_switch_{label}.json",
        f"results/{label}_bench_trace_switch.json",
        f"results/{label}_trace_switch.json",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _collect_bench_isolated(
    label: str,
    trace_path: str,
    benchmark_bin: str | None,
    bench_json_hint: str | None,
    benchmark_timeout_s: int,
) -> dict[str, float | int | str]:
    bench_json_path = _discover_bench_json_path(label, bench_json_hint)
    if bench_json_path is not None:
        try:
            parsed = _parse_replay_violation_bench_json(bench_json_path)
            return {
                "trace_label": label,
                "trace_path": trace_path,
                "status": "ok",
                "source": "json_file",
                "benchmark_json_path": bench_json_path,
                **parsed,
            }
        except Exception as exc:
            return {
                "trace_label": label,
                "trace_path": trace_path,
                "status": "warning",
                "source": "json_file",
                "benchmark_json_path": bench_json_path,
                "warning": f"Failed to parse benchmark JSON: {exc}",
                "trace_entries": 0,
                "violations": 0,
                "violation_rate": 0.0,
                "actual_ctx_switches": 0,
                "switch_p50_ns": 0.0,
                "switch_p99_ns": 0.0,
                "switch_min_ns": 0.0,
                "switch_max_ns": 0.0,
                "switch_p50_us": 0.0,
                "switch_p99_us": 0.0,
            }

    if benchmark_bin is not None:
        with tempfile.NamedTemporaryFile(
            prefix=f"bench_trace_switch_{label}_",
            suffix=".json",
            delete=False,
        ) as tmp:
            out_json = tmp.name
        cmd = [
            benchmark_bin,
            "--benchmark_filter=TraceSwitchBench/ReplayViolation",
            "--benchmark_out_format=json",
            f"--benchmark_out={out_json}",
        ]
        env = os.environ.copy()
        env["TRACE_CSV"] = trace_path
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=max(1, benchmark_timeout_s),
                check=False,
            )
            if proc.returncode == 0 and os.path.isfile(out_json):
                parsed = _parse_replay_violation_bench_json(out_json)
                return {
                    "trace_label": label,
                    "trace_path": trace_path,
                    "status": "ok",
                    "source": "benchmark_run",
                    "benchmark_bin": benchmark_bin,
                    "benchmark_json_path": out_json,
                    **parsed,
                }
            warning = (
                "Benchmark run failed: "
                f"return_code={proc.returncode}, stderr={proc.stderr.strip()}"
            )
            return {
                "trace_label": label,
                "trace_path": trace_path,
                "status": "warning",
                "source": "benchmark_run",
                "benchmark_bin": benchmark_bin,
                "warning": warning,
                "trace_entries": 0,
                "violations": 0,
                "violation_rate": 0.0,
                "actual_ctx_switches": 0,
                "switch_p50_ns": 0.0,
                "switch_p99_ns": 0.0,
                "switch_min_ns": 0.0,
                "switch_max_ns": 0.0,
                "switch_p50_us": 0.0,
                "switch_p99_us": 0.0,
            }
        except Exception as exc:
            return {
                "trace_label": label,
                "trace_path": trace_path,
                "status": "warning",
                "source": "benchmark_run",
                "benchmark_bin": benchmark_bin,
                "warning": f"Benchmark invocation error: {exc}",
                "trace_entries": 0,
                "violations": 0,
                "violation_rate": 0.0,
                "actual_ctx_switches": 0,
                "switch_p50_ns": 0.0,
                "switch_p99_ns": 0.0,
                "switch_min_ns": 0.0,
                "switch_max_ns": 0.0,
                "switch_p50_us": 0.0,
                "switch_p99_us": 0.0,
            }

    return {
        "trace_label": label,
        "trace_path": trace_path,
        "status": "missing",
        "source": "none",
        "warning": (
            "bench_trace_switch ReplayViolation data unavailable; "
            "provide --benchmark-bin or --bench-json-*"
        ),
        "trace_entries": 0,
        "violations": 0,
        "violation_rate": 0.0,
        "actual_ctx_switches": 0,
        "switch_p50_ns": 0.0,
        "switch_p99_ns": 0.0,
        "switch_min_ns": 0.0,
        "switch_max_ns": 0.0,
        "switch_p50_us": 0.0,
        "switch_p99_us": 0.0,
    }


def _measure_inline_trace(
    trace_path: str,
    steps: int,
    total_sms: int,
) -> dict[str, object]:
    runtime = _create_runtime()
    python_overheads_us: list[float] = []
    cpp_overheads_us: list[float] = []
    fallback_rows = 0
    fallback_switches = 0
    fallback_total_proxy_us = 0.0

    max_rows = max(1, steps) * 100
    sm_counts = _load_sm_counts(trace_path, max_rows)

    replayed_rows = 0
    for sm in sm_counts:
        effective_sm = total_sms - sm
        if effective_sm <= 0:
            continue
        replayed_rows += 1
        t0 = time.perf_counter_ns()
        prev_sm = runtime.activate_sm_for_thread(effective_sm)
        t1 = time.perf_counter_ns()
        activate_wall_us = (t1 - t0) / 1000.0
        python_overheads_us.append(activate_wall_us)

        cpp_stats = runtime.swap_stats()
        cpp_count = max(int(getattr(cpp_stats, "count", 0)), 0)
        cpp_total = max(float(getattr(cpp_stats, "total_overhead_us", 0)), 0.0)
        if cpp_count > 0:
            per_switch = cpp_total / cpp_count
            if cpp_total == 0.0 and activate_wall_us > 0.0:
                per_switch = activate_wall_us / cpp_count
                fallback_rows += 1
                fallback_switches += cpp_count
                fallback_total_proxy_us += activate_wall_us
            cpp_overheads_us.extend([per_switch] * cpp_count)

        runtime.deactivate_for_thread(prev_sm)

    runtime.close()
    return {
        "trace_path": trace_path,
        "replayed_rows": replayed_rows,
        "cpp_samples_us": cpp_overheads_us,
        "python_samples_us": python_overheads_us,
        "cpp_summary": _summarize_overhead_us(cpp_overheads_us),
        "python_summary": _summarize_overhead_us(python_overheads_us),
        "cpp_quantization_fallback": {
            "used": fallback_switches > 0,
            "rows": fallback_rows,
            "switches": fallback_switches,
            "proxy_total_overhead_us": fallback_total_proxy_us,
        },
    }


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _within_factor(ratio: float, factor: float) -> bool:
    if ratio <= 0 or factor <= 0:
        return False
    return max(ratio, 1.0 / ratio) <= factor


def _prefer_metric(primary: float, fallback: float) -> tuple[float, str]:
    if primary > 0.0:
        return primary, "p50_overhead_us"
    if fallback > 0.0:
        return fallback, "avg_overhead_us"
    return 0.0, "none"


def _to_int(raw: object, default: int = 0) -> int:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return default
    return default


def _resolve_trace_paths(args: TraceArgsProto) -> tuple[str, str]:
    default_with = "data/ldpc_trace_with_ctrl.csv"
    default_without = "data/ldpc_trace_without_ctrl.csv"

    legacy_trace = args.trace
    trace_with = args.trace_with
    trace_without = args.trace_without

    if trace_with is None and trace_without is None:
        if legacy_trace is not None:
            return legacy_trace, legacy_trace
        return default_with, default_without

    resolved_with = trace_with or legacy_trace or default_with
    resolved_without = trace_without or legacy_trace or default_without
    return resolved_with, resolved_without


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--trace",
        default=None,
        help="Legacy single-trace argument (compatibility mode)",
    )
    _ = parser.add_argument(
        "--trace-with",
        default=None,
        help="Trace path for with_ctrl replay",
    )
    _ = parser.add_argument(
        "--trace-without",
        default=None,
        help="Trace path for without_ctrl replay",
    )
    _ = parser.add_argument(
        "--output",
        default="results/swap_timing_validation.json",
    )
    _ = parser.add_argument("--steps", type=int, default=5)
    _ = parser.add_argument("--total-sms", type=int, default=48)
    _ = parser.add_argument(
        "--benchmark-bin",
        default=None,
        help="Path to bench_trace_switch binary",
    )
    _ = parser.add_argument(
        "--bench-json-with",
        default=None,
        help="Existing benchmark JSON for with_ctrl",
    )
    _ = parser.add_argument(
        "--bench-json-without",
        default=None,
        help="Existing benchmark JSON for without_ctrl",
    )
    _ = parser.add_argument("--benchmark-timeout-s", type=int, default=120)
    args = cast(ParsedArgs, cast(object, parser.parse_args()))

    trace_with, trace_without = _resolve_trace_paths(args)
    output = args.output
    steps = args.steps
    total_sms = args.total_sms

    print("=== Cross-validating swap timing ===")
    print(f"with_ctrl trace: {trace_with}")
    print(f"without_ctrl trace: {trace_without}")
    print(f"steps: {steps}")

    bench_bin = _resolve_benchmark_bin(args.benchmark_bin)
    bench_with = _collect_bench_isolated(
        label="with_ctrl",
        trace_path=trace_with,
        benchmark_bin=bench_bin,
        bench_json_hint=args.bench_json_with,
        benchmark_timeout_s=args.benchmark_timeout_s,
    )
    bench_without = _collect_bench_isolated(
        label="without_ctrl",
        trace_path=trace_without,
        benchmark_bin=bench_bin,
        bench_json_hint=args.bench_json_without,
        benchmark_timeout_s=args.benchmark_timeout_s,
    )

    inline_with = _measure_inline_trace(trace_with, steps, total_sms)
    inline_without = _measure_inline_trace(trace_without, steps, total_sms)

    inline_cpp_all = cast(list[float], inline_with["cpp_samples_us"]) + cast(
        list[float], inline_without["cpp_samples_us"]
    )
    inline_py_all = cast(list[float], inline_with["python_samples_us"]) + cast(
        list[float], inline_without["python_samples_us"]
    )

    inline_cpp_agg = _summarize_overhead_us(inline_cpp_all)
    inline_py_agg = _summarize_overhead_us(inline_py_all)

    bench_available = [
        bench_with,
        bench_without,
    ]
    bench_p50_us_values = [
        float(item["switch_p50_us"])
        for item in bench_available
        if item.get("status") == "ok" and float(item["switch_p50_us"]) > 0.0
    ]

    bench_p50_us_mean = (
        float(sum(bench_p50_us_values) / len(bench_p50_us_values))
        if bench_p50_us_values
        else 0.0
    )

    inline_cpp_metric_us, inline_cpp_metric_name = _prefer_metric(
        float(inline_cpp_agg["p50_overhead_us"]),
        float(inline_cpp_agg["avg_overhead_us"]),
    )
    inline_py_metric_us, inline_py_metric_name = _prefer_metric(
        float(inline_py_agg["p50_overhead_us"]),
        float(inline_py_agg["avg_overhead_us"]),
    )

    has_isolated_metric = bench_p50_us_mean > 0.0
    isolated_to_inline_ratio = (
        _safe_ratio(bench_p50_us_mean, inline_cpp_metric_us)
        if has_isolated_metric
        else 0.0
    )
    python_to_inline_ratio = _safe_ratio(inline_py_metric_us, inline_cpp_metric_us)

    isolated_within_2x = has_isolated_metric and _within_factor(
        isolated_to_inline_ratio,
        2.0,
    )
    isolated_discrepancy_over_5x = has_isolated_metric and (
        not _within_factor(isolated_to_inline_ratio, 5.0)
    )
    python_within_5x = python_to_inline_ratio > 0.0 and python_to_inline_ratio <= 5.0

    warning_note = ""
    if not has_isolated_metric:
        warning_note = (
            "bench_isolated ReplayViolation p50 is unavailable; "
            "inline numbers are authoritative"
        )
    elif isolated_discrepancy_over_5x:
        warning_note = (
            "Isolated vs inline C++ p50 discrepancy exceeds 5x; "
            "inline numbers are authoritative"
        )
        print(f"WARNING: {warning_note}")

    bench_status = "missing"
    if bench_with.get("status") == "ok" and bench_without.get("status") == "ok":
        bench_status = "ok"
    elif bench_with.get("status") == "ok" or bench_without.get("status") == "ok":
        bench_status = "partial"

    output_data: dict[str, object] = {
        "bench_isolated": {
            "with_ctrl": bench_with,
            "without_ctrl": bench_without,
            "aggregate": {
                "status": bench_status,
                "available_trace_count": len(bench_p50_us_values),
                "switch_p50_us_mean": bench_p50_us_mean,
                "switch_p50_ns_mean": bench_p50_us_mean * 1e3,
            },
        },
        "inline_cpp": {
            "with_ctrl": inline_with["cpp_summary"],
            "without_ctrl": inline_without["cpp_summary"],
            "aggregate": inline_cpp_agg,
            "quantization_fallback": {
                "with_ctrl": inline_with["cpp_quantization_fallback"],
                "without_ctrl": inline_without["cpp_quantization_fallback"],
                "aggregate": {
                    "used": bool(
                        cast(dict[str, object], inline_with["cpp_quantization_fallback"]).get("used", False)
                        or cast(dict[str, object], inline_without["cpp_quantization_fallback"]).get("used", False)
                    ),
                    "rows": _to_int(
                        cast(dict[str, object], inline_with["cpp_quantization_fallback"]).get("rows", 0)
                    )
                    + _to_int(
                        cast(dict[str, object], inline_without["cpp_quantization_fallback"]).get("rows", 0)
                    ),
                    "switches": _to_int(
                        cast(dict[str, object], inline_with["cpp_quantization_fallback"]).get("switches", 0)
                    )
                    + _to_int(
                        cast(dict[str, object], inline_without["cpp_quantization_fallback"]).get("switches", 0)
                    ),
                    "proxy_total_overhead_us": float(
                        _as_float(
                            cast(dict[str, object], inline_with["cpp_quantization_fallback"]).get(
                                "proxy_total_overhead_us", 0.0
                            )
                        )
                    )
                    + float(
                        _as_float(
                            cast(dict[str, object], inline_without["cpp_quantization_fallback"]).get(
                                "proxy_total_overhead_us", 0.0
                            )
                        )
                    ),
                },
            },
            "cross_validation": {
                "isolated_to_inline_cpp_p50_ratio": isolated_to_inline_ratio,
                "isolated_within_2x": isolated_within_2x,
                "isolated_discrepancy_over_5x": isolated_discrepancy_over_5x,
                "python_to_inline_cpp_p50_ratio": python_to_inline_ratio,
                "python_within_5x": python_within_5x,
                "isolated_ratio_check_skipped": not has_isolated_metric,
                "inline_cpp_ratio_metric_us": inline_cpp_metric_us,
                "inline_cpp_ratio_metric_name": inline_cpp_metric_name,
                "inline_python_ratio_metric_us": inline_py_metric_us,
                "inline_python_ratio_metric_name": inline_py_metric_name,
                "inline_authoritative": (
                    (not has_isolated_metric)
                    or isolated_discrepancy_over_5x
                ),
                "note": warning_note,
            },
            "metadata": {
                "steps": steps,
                "total_sms": total_sms,
            },
        },
        "inline_python": {
            "with_ctrl": inline_with["python_summary"],
            "without_ctrl": inline_without["python_summary"],
            "aggregate": inline_py_agg,
        },
    }

    out_dir = os.path.dirname(output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output}")
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
