#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportMissingTypeArgument=false, reportUnannotatedClassAttribute=false, reportRedeclaration=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark CLEAVE recovery breakdown"
    )
    parser.add_argument("--num-devices", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--model", type=str, default="opt-13b")
    parser.add_argument(
        "--failure-scenarios",
        type=str,
        default="1,3,5",
        help="Comma-separated failed-device counts",
    )
    parser.add_argument(
        "--detection-timeout-ms-range",
        type=str,
        default="50,200",
        help="Comma-separated [min,max] heartbeat detection timeout in ms",
    )
    parser.add_argument(
        "--solver-mode",
        type=str,
        default="mip",
        choices=["mip", "greedy"],
    )
    parser.add_argument("--mip-time-limit-s", type=float, default=20.0)
    parser.add_argument(
        "--coordinator-dl-mbps",
        type=float,
        default=12500.0,
        help="Coordinator egress used for re-dispatch phase (100Gbps = 12500 MB/s)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _import_baselines(repo_root: Path) -> tuple[Any, Any, Any] | None:
    baseline_root = repo_root.parent / "MobiCom26-Baselines"
    if str(baseline_root) not in sys.path:
        sys.path.insert(0, str(baseline_root))
    try:
        cfg_mod = __import__("baselines.core.config", fromlist=["ModelConfig"])
        strat_mod = __import__(
            "baselines.strategies.cleave_strategy",
            fromlist=["CleaveStrategy", "_DeviceRuntime"],
        )
        ModelConfig = getattr(cfg_mod, "ModelConfig")
        CleaveStrategy = getattr(strat_mod, "CleaveStrategy")
        _DeviceRuntime = getattr(strat_mod, "_DeviceRuntime")
        return ModelConfig, CleaveStrategy, _DeviceRuntime
    except Exception as exc:
        print(f"[WARN] Baselines import failed: {exc}")
        return None


def _model_config(cls_model: Any, model_name: str, batch_size: int, seq_len: int) -> Any:
    presets = {
        "opt-13b": {
            "num_layers": 40,
            "embedding_dim": 5120,
            "num_heads": 40,
            "d_ff": 20480,
        }
    }
    preset = presets.get(model_name.lower(), presets["opt-13b"])
    cfg = cls_model(
        model_name=model_name,
        seq_length=seq_len,
        max_seq_len=seq_len,
        num_layers=int(preset["num_layers"]),
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        d_ff=int(preset["d_ff"]),
    )
    setattr(cfg, "batch_size", int(batch_size))
    return cfg


def _make_devices(cls_device: Any, num_devices: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    devices: list[Any] = []
    for device_id in range(num_devices):
        devices.append(
            cls_device(
                device_id=device_id,
                flops=rng.uniform(5.0e12, 7.0e12),
                memory_bytes=8.0 * 1024.0 * 1024.0 * 1024.0,
                ul_bw_bytes_per_s=rng.uniform(5.0e6, 10.0e6),
                dl_bw_bytes_per_s=rng.uniform(10.0e6, 100.0e6),
                ul_lat_s=0.0,
                dl_lat_s=0.0,
            )
        )
    return devices


def _orphaned_keys(
    assignments: dict[tuple[int, int], list[tuple[int, float, float]]],
    failed_ids: set[int],
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for key, shards in assignments.items():
        for device_id, _alpha, _beta in shards:
            if int(device_id) in failed_ids:
                out.add(key)
                break
    return out


def _subset_dag(
    full_dag: list[list[tuple[int, int, int]]],
    keys: set[tuple[int, int]],
    failed_count: int,
    total_devices: int,
) -> list[list[tuple[int, int, int]]]:
    fraction = max(0.001, float(failed_count) / max(1, float(total_devices)))
    out: list[list[tuple[int, int, int]]] = []
    for level_idx, level in enumerate(full_dag):
        for gemm_idx, (m, n, q) in enumerate(level):
            if (level_idx, gemm_idx) in keys:
                m_s = max(1, int(float(m) * fraction))
                q_s = max(1, int(float(q) * fraction))
                out.append([(m_s, n, q_s)])
    return out


def _phase_detection_ms(min_ms: float, max_ms: float, rng: random.Random) -> float:
    low = max(1.0, min(float(min_ms), float(max_ms)))
    high = max(low, max(float(min_ms), float(max_ms)))
    return rng.uniform(low, high)


def _phase_redispatch_ms(
    assignments: dict[tuple[int, int], list[tuple[int, float, float]]],
    orphaned: set[tuple[int, int]],
    byte_size: float,
    coordinator_dl_bytes_s: float,
) -> float:
    total_bytes = 0.0
    for key in orphaned:
        for _device_id, alpha, beta in assignments.get(key, []):
            total_bytes += byte_size * float(alpha) * float(beta)
    if coordinator_dl_bytes_s <= 0.0:
        return 0.0
    num_orphaned = max(1, len(orphaned))
    per_shard_bytes = total_bytes / float(num_orphaned)
    max_shard_bytes = per_shard_bytes * min(3.0, float(num_orphaned))
    return (max_shard_bytes / coordinator_dl_bytes_s) * 1000.0


def _phase_recompute_ms(
    subset_dag: list[list[tuple[int, int, int]]],
    survivors: list[Any],
) -> float:
    if not subset_dag or not survivors:
        return 0.0
    total_ops = 0.0
    for level in subset_dag:
        for m_dim, n_dim, q_dim in level:
            total_ops += 2.0 * float(m_dim) * float(n_dim) * float(q_dim)
    # Recompute is distributed across survivors, so divide by aggregate FLOPS
    aggregate_flops = sum(float(d.flops) for d in survivors)
    return (total_ops / max(1.0, aggregate_flops)) * 1000.0


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    output_json = (
        args.output_json
        if args.output_json is not None
        else repo_root
        / "results"
        / "microbenchmarks"
        / "recovery_breakdown.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    failure_scenarios = [
        int(x.strip())
        for x in str(args.failure_scenarios).split(",")
        if x.strip()
    ]
    timeout_parts = [
        float(x.strip())
        for x in str(args.detection_timeout_ms_range).split(",")
        if x.strip()
    ]
    if len(timeout_parts) != 2:
        raise ValueError("--detection-timeout-ms-range must contain min,max")
    tmin, tmax = timeout_parts

    imports = _import_baselines(repo_root)
    if imports is None:
        rng = random.Random(args.seed)
        scenarios: dict[str, Any] = {}
        for fail_n in failure_scenarios:
            detection_ms = _phase_detection_ms(tmin, tmax, rng)
            resolve_ms = 30.0 + 18.0 * float(fail_n)
            redispatch_ms = 8.0 + 10.0 * float(fail_n)
            recompute_ms = 40.0 + 22.0 * float(fail_n)
            total_ms = detection_ms + resolve_ms + redispatch_ms + recompute_ms
            scenarios[str(fail_n)] = {
                "failed_devices": int(fail_n),
                "phases_ms": {
                    "detection": detection_ms,
                    "re_solve": resolve_ms,
                    "re_dispatch": redispatch_ms,
                    "recompute": recompute_ms,
                },
                "total_recovery_ms": total_ms,
                "solver": {"mode": str(args.solver_mode), "source": "estimated"},
            }
        payload = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "script": Path(__file__).name,
                "status": "baselines_unavailable",
            },
            "scenarios": scenarios,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] wrote {output_json}")
        return 0

    cls_model, cls_strategy, cls_device = imports
    strategy = cls_strategy(
        prefer_mip=True,
        mip_time_limit_s=float(args.mip_time_limit_s),
    )
    model_cfg = _model_config(
        cls_model=cls_model,
        model_name=args.model,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
    )
    devices = _make_devices(
        cls_device=cls_device,
        num_devices=int(args.num_devices),
        seed=int(args.seed) + 7,
    )
    full_dag = strategy._resolve_gemm_dag(model_cfg)
    full_assignments, _ = strategy.solve_greedy(full_dag, devices)

    rng = random.Random(args.seed)
    scenarios: dict[str, Any] = {}
    coordinator_dl_bytes_s = float(args.coordinator_dl_mbps) * 1024.0 * 1024.0

    for fail_n in failure_scenarios:
        failed = set(
            rng.sample(
                range(int(args.num_devices)),
                k=min(max(1, int(fail_n)), int(args.num_devices)),
            )
        )
        survivors = [d for d in devices if int(d.device_id) not in failed]
        orphaned = _orphaned_keys(full_assignments, failed)
        orphaned_dag = _subset_dag(full_dag, orphaned, len(failed), int(args.num_devices))
        if not orphaned_dag:
            orphaned_dag = [[shape] for level in full_dag for shape in level][:6]

        detection_ms = _phase_detection_ms(tmin, tmax, rng)

        re_solve_ms: float
        solver_source = "measured"
        t0 = time.perf_counter()
        if args.solver_mode == "mip":
            try:
                mip_result = strategy.solve_mip(orphaned_dag, survivors)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if mip_result is None:
                    solver_source = "estimated"
                    re_solve_ms = max(5.0, 12.0 * float(len(orphaned_dag)) / 10.0)
                else:
                    re_solve_ms = elapsed_ms
            except Exception:
                solver_source = "estimated"
                re_solve_ms = max(5.0, 12.0 * float(len(orphaned_dag)) / 10.0)
        else:
            _ = strategy.solve_greedy(orphaned_dag, survivors)
            re_solve_ms = (time.perf_counter() - t0) * 1000.0

        re_dispatch_ms = _phase_redispatch_ms(
            assignments=full_assignments,
            orphaned=orphaned,
            byte_size=float(strategy.byte_size),
            coordinator_dl_bytes_s=coordinator_dl_bytes_s,
        )
        recompute_ms = _phase_recompute_ms(orphaned_dag, survivors)
        total_ms = detection_ms + re_solve_ms + re_dispatch_ms + recompute_ms

        scenarios[str(fail_n)] = {
            "failed_devices": int(fail_n),
            "failed_device_ids": sorted(int(x) for x in failed),
            "orphaned_gemms": len(orphaned_dag),
            "phases_ms": {
                "detection": detection_ms,
                "re_solve": re_solve_ms,
                "re_dispatch": re_dispatch_ms,
                "recompute": recompute_ms,
            },
            "total_recovery_ms": total_ms,
            "solver": {
                "mode": str(args.solver_mode),
                "source": solver_source,
            },
        }

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": Path(__file__).name,
            "model": args.model,
            "num_devices": int(args.num_devices),
            "batch_size": int(args.batch_size),
            "seq_len": int(args.seq_len),
            "failure_scenarios": failure_scenarios,
            "detection_timeout_ms_range": [float(tmin), float(tmax)],
            "solver_mode": str(args.solver_mode),
            "coordinator_dl_mbps": float(args.coordinator_dl_mbps),
        },
        "scenarios": scenarios,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
