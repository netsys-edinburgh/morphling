#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import baselines.strategies.cleave_strategy as _cleave_mod
import baselines.strategies.dtfm_strategy as _dtfm_mod
from baselines.core.config import ModelConfig
from baselines.strategies import (
    AlpaStrategy,
    AsteroidStrategy,
    CleaveStrategy,
    ConfidentStrategy,
    DTFMStrategy,
)

from morphling.evaluation.plan_dispatcher import (
    DispatchManifest,
    gemm_dag_from_model_config,
    plan_to_manifest,
)
from morphling.evaluation.topology_bridge import (
    fleet_to_topology,
    load_fleet_json,
)
from scripts.compute_batch_runtime import (
    BatchRuntimeResult,
    compute_batch_runtime,
    parse_vtime_log,
)

_dtfm_mod._HAS_GCMA = False

LOGGER = logging.getLogger("run_baseline_comparison")


class FastCleaveStrategy(CleaveStrategy):
    """Cleave wrapper with linear-time topology aggregation."""

    def _extract_device_runtime(self, topology):
        incoming_bw_sum: dict[int, float] = {}
        incoming_bw_count: dict[int, int] = {}
        outgoing_bw_sum: dict[int, float] = {}
        outgoing_bw_count: dict[int, int] = {}
        incoming_lat_sum: dict[int, float] = {}
        incoming_lat_count: dict[int, int] = {}
        outgoing_lat_sum: dict[int, float] = {}
        outgoing_lat_count: dict[int, int] = {}

        for (src, dst), value in topology.bandwidths.items():
            bw = float(value)
            outgoing_bw_sum[src] = outgoing_bw_sum.get(src, 0.0) + bw
            outgoing_bw_count[src] = outgoing_bw_count.get(src, 0) + 1
            incoming_bw_sum[dst] = incoming_bw_sum.get(dst, 0.0) + bw
            incoming_bw_count[dst] = incoming_bw_count.get(dst, 0) + 1

        for (src, dst), value in topology.latencies.items():
            lat = float(value)
            outgoing_lat_sum[src] = outgoing_lat_sum.get(src, 0.0) + lat
            outgoing_lat_count[src] = outgoing_lat_count.get(src, 0) + 1
            incoming_lat_sum[dst] = incoming_lat_sum.get(dst, 0.0) + lat
            incoming_lat_count[dst] = incoming_lat_count.get(dst, 0) + 1

        runtimes = []
        for spec in topology.device_specs:
            device_id = int(spec.device_id)
            dl_bw_mbps = incoming_bw_sum.get(device_id, 100.0) / max(
                1, incoming_bw_count.get(device_id, 0)
            )
            ul_bw_mbps = outgoing_bw_sum.get(device_id, 100.0) / max(
                1, outgoing_bw_count.get(device_id, 0)
            )
            dl_lat_ms = incoming_lat_sum.get(device_id, 0.1) / max(
                1, incoming_lat_count.get(device_id, 0)
            )
            ul_lat_ms = outgoing_lat_sum.get(device_id, 0.1) / max(
                1, outgoing_lat_count.get(device_id, 0)
            )

            runtimes.append(
                _cleave_mod._DeviceRuntime(
                    device_id=device_id,
                    flops=self._normalize_flops(spec.compute_capacity),
                    memory_bytes=max(1.0, spec.memory_budget_mb) * 1024.0 * 1024.0,
                    ul_bw_bytes_per_s=max(1e-9, ul_bw_mbps) * 1024.0 * 1024.0,
                    dl_bw_bytes_per_s=max(1e-9, dl_bw_mbps) * 1024.0 * 1024.0,
                    ul_lat_s=max(0.0, ul_lat_ms) / 1000.0,
                    dl_lat_s=max(0.0, dl_lat_ms) / 1000.0,
                )
            )
        return runtimes


class FastDTFMStrategy(DTFMStrategy):
    """DTFM wrapper that caches repeated stage-boundary communication costs."""

    def _dp_partition(
        self,
        model_config,
        topology,
        device_groups,
        profiler,
    ):
        num_layers = max(1, model_config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = min(len(stage_ids), num_layers)
        spec_by_id = {
            spec.device_id: spec
            for spec in topology.device_specs
        }
        if num_stages <= 1:
            group = device_groups.get(stage_ids[0] if stage_ids else 0, [])
            if not group:
                return [], 0.0
            worst_total = 0.0
            for did in group:
                sp = spec_by_id.get(did, _dtfm_mod.DeviceConfig(device_id=did))
                total = sum(
                    self._layer_time(li, did, sp, model_config, profiler)
                    for li in range(num_layers)
                )
                worst_total = max(worst_total, total)
            return [], worst_total

        stage_prefix: list[list[float]] = []
        for stage_idx in range(num_stages):
            stage = stage_ids[stage_idx]
            group = device_groups[stage]
            rep_id = group[0]
            rep_time = float("-inf")
            for did in group:
                sp = spec_by_id.get(did, _dtfm_mod.DeviceConfig(device_id=did))
                t = sum(
                    self._layer_time(li, did, sp, model_config, profiler)
                    for li in range(num_layers)
                )
                if t > rep_time:
                    rep_time = t
                    rep_id = did
            spec = spec_by_id.get(rep_id, _dtfm_mod.DeviceConfig(device_id=rep_id))
            prefix = [0.0]
            for layer_idx in range(num_layers):
                layer_t = self._layer_time(
                    layer_idx,
                    rep_id,
                    spec,
                    model_config,
                    profiler,
                )
                prefix.append(prefix[-1] + layer_t)
            stage_prefix.append(prefix)

        boundary_costs: list[float] = []
        for stage_idx in range(1, num_stages):
            boundary_costs.append(
                self._boundary_comm_time(
                    boundary_layer=stage_idx - 1,
                    left_group=device_groups[stage_ids[stage_idx - 1]],
                    right_group=device_groups[stage_ids[stage_idx]],
                    model_config=model_config,
                    topology=topology,
                    profiler=profiler,
                )
            )

        dp = [
            [float("inf")] * num_stages
            for _ in range(num_layers)
        ]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(stage_idx: int, start: int, end: int) -> float:
            prefix = stage_prefix[stage_idx]
            return prefix[end + 1] - prefix[start]

        for i in range(num_layers):
            dp[i][0] = range_cost(0, 0, i)

        for stage_idx in range(1, num_stages):
            comm = boundary_costs[stage_idx - 1]
            for end in range(stage_idx, num_layers):
                for cut in range(stage_idx - 1, end):
                    comp = range_cost(stage_idx, cut + 1, end)
                    candidate = max(dp[cut][stage_idx - 1], comp + comm)
                    if candidate < dp[end][stage_idx]:
                        dp[end][stage_idx] = candidate
                        split[end][stage_idx] = cut

        points: list[int] = []
        end = num_layers - 1
        stage_idx = num_stages - 1
        while stage_idx > 0:
            cut = split[end][stage_idx]
            if cut < 0:
                break
            points.append(cut)
            end = cut
            stage_idx -= 1
        points.reverse()
        bottleneck = dp[num_layers - 1][num_stages - 1]
        if bottleneck == float("inf"):
            points = self._fallback_points(num_layers, num_stages)
            bottleneck = 0.0
        return points, bottleneck


class FastAsteroidStrategy(AsteroidStrategy):
    """Asteroid wrapper that avoids the exact DP path at the 32-device cliff."""

    _EXACT_DP_DEVICE_LIMIT = 31

    def create_plan(
        self,
        model_config,
        device_topology,
        profiler=None,
    ):
        topology = self._normalize_topology(device_topology)
        exec_profiles = self._build_exec_profiles(
            model_config,
            topology,
            profiler,
        )
        max_stages = min(
            max(1, self.num_stages + 2),
            len(topology.device_specs),
            max(1, model_config.num_layers),
        )
        num_devices = len(topology.device_specs)

        best_plan = None
        if num_devices <= self._EXACT_DP_DEVICE_LIMIT:
            for stages in range(1, max_stages + 1):
                candidate = self._dp_plan(
                    model_config,
                    topology,
                    exec_profiles,
                    stages,
                )
                if candidate is None:
                    continue
                if (
                    best_plan is None
                    or candidate.estimated_latency_ms < best_plan.estimated_latency_ms
                ):
                    best_plan = candidate

        if best_plan is None:
            best_plan = self._fallback_plan(
                model_config,
                topology,
                exec_profiles,
            )
        LOGGER.info(
            "Fast asteroid plan points=%s latency=%.2fms",
            best_plan.partition_points,
            best_plan.estimated_latency_ms,
        )
        return best_plan

SUPPORTED_BASELINES = {
    "cleave",
    "dtfm",
    "asteroid",
    "confident",
    "alpa",
}

MODEL_PRESETS: dict[str, dict[str, int]] = {
    "facebook/opt-125m": {
        "num_layers": 12,
        "embedding_dim": 768,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "opt-125m": {
        "num_layers": 12,
        "embedding_dim": 768,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "facebook/opt-1.3b": {
        "num_layers": 24,
        "embedding_dim": 2048,
        "num_heads": 32,
        "d_ff": 8192,
    },
    "opt-1.3b": {
        "num_layers": 24,
        "embedding_dim": 2048,
        "num_heads": 32,
        "d_ff": 8192,
    },
    "facebook/opt-13b": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 20480,
    },
    "opt-13b": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 20480,
    },
    "meta-llama/llama-2-7b-hf": {
        "num_layers": 32,
        "embedding_dim": 4096,
        "num_heads": 32,
        "d_ff": 11008,
    },
    "llama2-7b": {
        "num_layers": 32,
        "embedding_dim": 4096,
        "num_heads": 32,
        "d_ff": 11008,
    },
    "llama-2-7b": {
        "num_layers": 32,
        "embedding_dim": 4096,
        "num_heads": 32,
        "d_ff": 11008,
    },
    "facebook/opt-30b": {
        "num_layers": 48,
        "embedding_dim": 7168,
        "num_heads": 56,
        "d_ff": 28672,
    },
    "opt-30b": {
        "num_layers": 48,
        "embedding_dim": 7168,
        "num_heads": 56,
        "d_ff": 28672,
    },
    "facebook/opt-66b": {
        "num_layers": 64,
        "embedding_dim": 9216,
        "num_heads": 72,
        "d_ff": 36864,
    },
    "opt-66b": {
        "num_layers": 64,
        "embedding_dim": 9216,
        "num_heads": 72,
        "d_ff": 36864,
    },
    "meta-llama/llama-2-13b-hf": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 13824,
    },
    "llama2-13b": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 13824,
    },
    "llama-2-13b": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 13824,
    },
    "meta-llama/llama-2-70b-hf": {
        "num_layers": 80,
        "embedding_dim": 8192,
        "num_heads": 64,
        "d_ff": 28672,
    },
    "llama2-70b": {
        "num_layers": 80,
        "embedding_dim": 8192,
        "num_heads": 64,
        "d_ff": 28672,
    },
    "llama-2-70b": {
        "num_layers": 80,
        "embedding_dim": 8192,
        "num_heads": 64,
        "d_ff": 28672,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline planners end-to-end, generate dispatch manifests, "
            "and compare per-batch runtime."
        )
    )
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--num-devices", required=True, type=int)
    parser.add_argument("--device-config", default=None, type=str)
    parser.add_argument(
        "--baselines",
        default="cleave,dtfm,asteroid,confident,alpa",
        type=str,
    )
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--vtime-log", default=None, type=str)
    parser.add_argument("--micro-batch-size", default=1, type=int)
    parser.add_argument("--num-microbatches", default=None, type=int)
    parser.add_argument("--bytes-per-element", default=2.0, type=float)
    parser.add_argument("--allreduce-bandwidth-bps", default=None, type=float)
    parser.add_argument("--log-level", default="INFO", type=str)
    return parser.parse_args()


def _normalize_model_name(model: str) -> str:
    return model.strip().lower()


def _model_defaults(model_name: str) -> dict[str, int]:
    normalized = _normalize_model_name(model_name)
    preset = MODEL_PRESETS.get(normalized)
    if preset is not None:
        return dict(preset)

    try:
        from morphling.utils.hfparser import parse_model_meta

        meta = parse_model_meta(model_name)
        return {
            "num_layers": int(meta["n_layer"]),
            "embedding_dim": int(meta["d_model"]),
            "num_heads": int(meta["n_head"]),
            "d_ff": int(meta["d_ffn"]),
        }
    except Exception as exc:
        LOGGER.warning(
            "Failed to fetch model metadata for %s (%s). Falling back to "
            "generic defaults.",
            model_name,
            exc,
        )
        return {
            "num_layers": 12,
            "embedding_dim": 768,
            "num_heads": 12,
            "d_ff": 3072,
        }


def _build_model_config(
    model_name: str,
    batch_size: int,
    seq_len: int,
) -> ModelConfig:
    defaults = _model_defaults(model_name)
    model_config = ModelConfig(
        model_name=model_name,
        seq_length=seq_len,
        max_seq_len=seq_len,
        num_layers=int(defaults["num_layers"]),
        embedding_dim=int(defaults["embedding_dim"]),
        num_heads=int(defaults["num_heads"]),
        d_ff=int(defaults["d_ff"]),
    )
    setattr(model_config, "batch_size", batch_size)
    return model_config


def _generate_default_fleet(
    num_devices: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    memory_bytes = 8 * 1024 * 1024 * 1024
    fleet: list[dict[str, Any]] = []

    for rank in range(num_devices):
        fleet.append(
            {
                "rank": rank,
                "flops": int(rng.uniform(5.0e12, 7.0e12)),
                "memory": memory_bytes,
                "ul_bw": float(rng.uniform(5.0e6, 10.0e6)),
                "dl_bw": float(rng.uniform(10.0e6, 100.0e6)),
                "ul_lat": 0.0,
                "dl_lat": 0.0,
            }
        )
    return fleet


def _load_or_generate_fleet(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.device_config:
        raw = load_fleet_json(args.device_config)
        if not isinstance(raw, list):
            raise ValueError(
                "Device fleet JSON must be a list of device records"
            )
        fleet = [item for item in raw if isinstance(item, dict)]
        if len(fleet) < args.num_devices:
            raise ValueError(
                "Device fleet has fewer devices than --num-devices: "
                f"{len(fleet)} < {args.num_devices}"
            )
        if len(fleet) > args.num_devices:
            LOGGER.warning(
                "Device fleet has %d devices; truncating to first %d.",
                len(fleet),
                args.num_devices,
            )
            fleet = sorted(fleet, key=lambda x: int(x.get("rank", 0)))[
                : args.num_devices
            ]
        return fleet

    LOGGER.info(
        "No --device-config provided; generating synthetic fleet with seed=%d.",
        args.seed,
    )
    return _generate_default_fleet(args.num_devices, args.seed)


def _fleet_device_profiles(
    fleet: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    profiles: dict[int, dict[str, float]] = {}
    for device in fleet:
        rank = int(device["rank"])
        profiles[rank] = {
            "flops": float(device["flops"]),
            "ul_bw": float(device["ul_bw"]),
            "dl_bw": float(device["dl_bw"]),
            "ul_lat": float(device.get("ul_lat", 0.0)),
            "dl_lat": float(device.get("dl_lat", 0.0)),
        }
    return profiles


def _gemm_tuple_dag(
    dag: list[list[dict[str, Any]]],
) -> list[list[tuple[int, int, int]]]:
    return [
        [(int(item["m"]), int(item["n"]), int(item["q"])) for item in level]
        for level in dag
    ]


def _best_pp_dp(num_devices: int, num_layers: int) -> tuple:
    """Find (pp_size, dp_size) factorization that fits within num_layers."""
    best_pp, best_dp = 1, num_devices
    for pp in range(min(num_devices, num_layers), 0, -1):
        if num_devices % pp == 0:
            best_pp, best_dp = pp, num_devices // pp
            break
    return best_pp, best_dp


def _build_strategy(
    baseline: str,
    num_devices: int,
    num_layers: int,
    batch_size: int,
    micro_batch_size: int,
    num_microbatches: int,
):
    pp_size, dp_size = _best_pp_dp(num_devices, num_layers)
    if baseline == "cleave":
        return FastCleaveStrategy()
    if baseline == "dtfm":
        return FastDTFMStrategy(
            pp_size=pp_size,
            dp_size=dp_size,
            population_size=10,
            gcma_trails=50,
            global_batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )
    if baseline == "asteroid":
        return FastAsteroidStrategy(
            num_stages=pp_size,
            micro_batch_size=micro_batch_size,
            num_microbatches=num_microbatches,
        )
    if baseline == "confident":
        return ConfidentStrategy(pp_size=pp_size, dp_size=dp_size)
    if baseline == "alpa":
        return AlpaStrategy(
            num_microbatches=num_microbatches,
            micro_batch_size=micro_batch_size,
        )
    raise ValueError(f"Unsupported baseline: {baseline}")


def _run_single_baseline(
    baseline: str,
    model_config: ModelConfig,
    gemm_dag: list[list[dict[str, Any]]],
    topology: Any,
    device_profiles: dict[int, dict[str, float]],
    vtime_events: list[Any],
    args: argparse.Namespace,
    manifests_dir: Path,
    runtimes_dir: Path,
) -> tuple[DispatchManifest, BatchRuntimeResult]:
    num_microbatches = args.num_microbatches
    if num_microbatches is None:
        num_microbatches = max(
            1, args.batch_size // max(1, args.micro_batch_size)
        )

    strategy = _build_strategy(
        baseline=baseline,
        num_devices=args.num_devices,
        num_layers=model_config.num_layers,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_microbatches=num_microbatches,
    )

    import signal

    class _PlannerTimeout(Exception):
        pass

    def _alarm_handler(signum, frame):
        raise _PlannerTimeout

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(60)
    try:
        if baseline == "cleave":
            plan = strategy.create_plan(
                model_config,
                topology,
                profiler=None,
                gemm_dag=_gemm_tuple_dag(gemm_dag),
            )
        else:
            plan = strategy.create_plan(model_config, topology, profiler=None)
    except _PlannerTimeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        raise RuntimeError(f"{baseline} planner timed out after 60s")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    manifest = plan_to_manifest(
        plan=plan,
        gemm_dag=gemm_dag,
        baseline_type=baseline,
        model_config=model_config,
    )

    manifest_path = manifests_dir / f"{baseline}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_json(), indent=2),
        encoding="utf-8",
    )

    runtime_result = compute_batch_runtime(
        vtime_events=vtime_events,
        manifest=manifest.to_json(),
        baseline_type=baseline,
        pp_size=max(1, len(getattr(plan, "partition_points", []) or []) + 1),
        num_microbatches=num_microbatches,
        optimizer_tail_ms=float(getattr(plan, "optimizer_tail_ms", 0.0)),
        device_profiles=device_profiles,
        bytes_per_element=float(args.bytes_per_element),
        allreduce_bandwidth_bps=args.allreduce_bandwidth_bps,
    )

    runtime_payload = asdict(runtime_result)
    runtime_payload["mode"] = "vtime" if args.vtime_log else "analytical"
    runtime_path = runtimes_dir / f"{baseline}_runtime.json"
    runtime_path.write_text(
        json.dumps(runtime_payload, indent=2), encoding="utf-8"
    )

    return manifest, runtime_result


def _build_summary_table(
    model_name: str,
    num_devices: int,
    baselines: list[str],
    runtime_results: dict[str, BatchRuntimeResult],
    normalized: dict[str, float | None],
) -> str:
    title = f"Baseline Comparison: {model_name}, {num_devices} devices"
    sep = "─" * max(len(title), 76)
    lines = [title, sep]
    lines.append(
        "Baseline    Total(ms)  Compute(ms)  Network(ms)  Bubble(ms)  "
        "Normalized"
    )

    for baseline in baselines:
        result = runtime_results.get(baseline)
        if result is None:
            lines.append(f"{baseline:<10} {'FAILED':>10}")
            continue
        norm = normalized.get(baseline)
        norm_text = "N/A" if norm is None else f"{norm:.2f}x"
        lines.append(
            f"{baseline:<10}"
            f"{result.total_runtime_ms:>10.1f}"
            f"{result.compute_time_ms:>13.1f}"
            f"{result.network_time_ms:>13.1f}"
            f"{result.pipeline_bubble_ms:>12.1f}"
            f"{norm_text:>12}"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    raw_baselines = [item.strip().lower() for item in args.baselines.split(",")]
    baselines = [item for item in raw_baselines if item]
    for baseline in baselines:
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(
                f"Unsupported baseline '{baseline}'. "
                f"Supported: {sorted(SUPPORTED_BASELINES)}"
            )

    output_dir = Path(args.output_dir)
    manifests_dir = output_dir / "manifests"
    runtimes_dir = output_dir / "runtimes"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    runtimes_dir.mkdir(parents=True, exist_ok=True)

    fleet = _load_or_generate_fleet(args)
    topology = fleet_to_topology(fleet)
    device_profiles = _fleet_device_profiles(fleet)

    if not args.device_config:
        generated_fleet_path = output_dir / "generated_device_fleet.json"
        generated_fleet_path.write_text(
            json.dumps(fleet, indent=2), encoding="utf-8"
        )

    model_config = _build_model_config(
        model_name=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    gemm_dag = gemm_dag_from_model_config(model_config)
    if not gemm_dag:
        raise RuntimeError("GEMM DAG generation failed: empty DAG")

    vtime_events: list[Any] = []
    if args.vtime_log:
        vtime_events = parse_vtime_log(args.vtime_log)
        LOGGER.info(
            "Loaded %d VTIME events from %s", len(vtime_events), args.vtime_log
        )

    runtime_results: dict[str, BatchRuntimeResult] = {}
    failures: dict[str, str] = {}

    for baseline in baselines:
        try:
            LOGGER.info("Running baseline planner: %s", baseline)
            _manifest, runtime_result = _run_single_baseline(
                baseline=baseline,
                model_config=model_config,
                gemm_dag=gemm_dag,
                topology=topology,
                device_profiles=device_profiles,
                vtime_events=vtime_events,
                args=args,
                manifests_dir=manifests_dir,
                runtimes_dir=runtimes_dir,
            )
            runtime_results[baseline] = runtime_result
        except Exception as exc:
            message = str(exc)
            failures[baseline] = message
            LOGGER.warning(
                "Baseline %s failed, skipping: %s", baseline, message
            )

    baselines_payload: dict[str, dict[str, Any]] = {}
    for baseline, result in runtime_results.items():
        baselines_payload[baseline] = asdict(result)

    cleave_runtime = runtime_results.get("cleave")
    normalized: dict[str, float | None] = {}
    if cleave_runtime is not None and cleave_runtime.total_runtime_ms > 0:
        cleave_total = cleave_runtime.total_runtime_ms
        for baseline, result in runtime_results.items():
            normalized[baseline] = result.total_runtime_ms / cleave_total
    else:
        for baseline in runtime_results:
            normalized[baseline] = None
        if "cleave" in baselines and "cleave" not in runtime_results:
            LOGGER.warning(
                "CLEAVE did not complete; normalized metrics set to null."
            )

    summary_payload: dict[str, Any] = {
        "model": args.model,
        "num_devices": args.num_devices,
        "baselines": baselines_payload,
        "normalized": normalized,
    }
    if failures:
        summary_payload["failures"] = failures

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    summary_table = _build_summary_table(
        model_name=args.model,
        num_devices=args.num_devices,
        baselines=baselines,
        runtime_results=runtime_results,
        normalized=normalized,
    )
    summary_table_path = output_dir / "summary_table.txt"
    summary_table_path.write_text(summary_table + "\n", encoding="utf-8")

    print(summary_table)
    print(f"\nSaved summary JSON: {summary_json_path}")
    print(f"Saved summary table: {summary_table_path}")
    print(f"Saved manifests dir: {manifests_dir}")
    print(f"Saved runtimes dir: {runtimes_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
