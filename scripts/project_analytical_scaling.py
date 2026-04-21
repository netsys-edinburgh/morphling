#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false, reportImplicitRelativeImport=false, reportImplicitStringConcatenation=false
"""D2: Analytical scaling projections with calibration corrections.

For each (num_devices, model, method) triple in the cross-product,
runs the analytical cost model and applies calibration correction
factors from D1.

Also supports --validate to cross-check against existing
device_scaling results (D4).
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    gemm_dag_from_model_config,
    plan_to_manifest,
)
from morphling.evaluation.topology_bridge import fleet_to_topology
from scripts.compute_batch_runtime import (
    BatchRuntimeResult,
    compute_batch_runtime,
)

_dtfm_mod._HAS_GCMA = False

LOG = logging.getLogger("project_analytical_scaling")

REPO = Path(__file__).resolve().parent.parent
CALIBRATION_PATH = REPO / "results/analytical_scaling/calibration.json"
FLEET_PATH = (
    REPO / "results/device_scaling/opt-125m/n64" / "generated_device_fleet.json"
)

FLEET_PROFILES = {
    "mild": {
        "flops": (5e12, 7e12),
        "ul_bw": (5e6, 10e6),
        "dl_bw": (14e6, 100e6),
    },
    "moderate": {
        "flops": (2e12, 10e12),
        "ul_bw": (2e6, 20e6),
        "dl_bw": (5e6, 200e6),
    },
    "extreme": {
        "flops": (1e12, 20e12),
        "ul_bw": (1e6, 50e6),
        "dl_bw": (1e6, 500e6),
    },
}
OUTPUT_DIR = REPO / "results/analytical_scaling"
PROJECTIONS_PATH = OUTPUT_DIR / "projections.json"
VALIDATION_PATH = OUTPUT_DIR / "validation.json"
DEVICE_SCALING_DIR = REPO / "results/device_scaling"

DEVICE_COUNTS = [64, 128, 256, 512, 1024, 2048]
METHODS = ["cleave", "dtfm", "asteroid", "confident", "alpa"]

# Authoritative model configs (from run_baseline_comparison.py)
MODEL_CONFIGS = {
    "opt-125m": {
        "num_layers": 12,
        "embedding_dim": 768,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "opt-1.3b": {
        "num_layers": 24,
        "embedding_dim": 2048,
        "num_heads": 32,
        "d_ff": 8192,
    },
    "llama2-7b": {
        "num_layers": 32,
        "embedding_dim": 4096,
        "num_heads": 32,
        "d_ff": 11008,
    },
    "opt-13b": {
        "num_layers": 40,
        "embedding_dim": 5120,
        "num_heads": 40,
        "d_ff": 20480,
    },
    "opt-30b": {
        "num_layers": 48,
        "embedding_dim": 7168,
        "num_heads": 56,
        "d_ff": 28672,
    },
    "opt-66b": {
        "num_layers": 64,
        "embedding_dim": 9216,
        "num_heads": 72,
        "d_ff": 36864,
    },
}

BATCH_SIZE = 16
MICRO_BATCH_SIZE = 1
NUM_MICROBATCHES = 16
SEQ_LEN = 512
BYTES_PER_ELEMENT = 2.0


# -- fleet helpers --


def _load_base_fleet(
    profile: str | None = None, seed: int = 42
) -> list[dict[str, Any]]:
    if profile is None:
        with open(FLEET_PATH) as f:
            return json.load(f)
    return _generate_fleet(64, profile, seed)


def _generate_fleet(n: int, profile: str, seed: int) -> list[dict[str, Any]]:
    import random

    rng = random.Random(seed)
    cfg = FLEET_PROFILES[profile]
    fleet = []
    for i in range(n):
        fleet.append(
            {
                "rank": i,
                "flops": int(rng.uniform(*cfg["flops"])),
                "memory": 8 * 1024 * 1024 * 1024,
                "ul_bw": float(rng.uniform(*cfg["ul_bw"])),
                "dl_bw": float(rng.uniform(*cfg["dl_bw"])),
                "ul_lat": 0.0,
                "dl_lat": 0.0,
            }
        )
    return fleet


def _repeat_fleet(base: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Build an N-device fleet by repeating the base fleet."""
    fleet = []
    for i in range(n):
        src = base[i % len(base)]
        dev = dict(src)
        dev["rank"] = i
        fleet.append(dev)
    return fleet


def _fleet_profiles(
    fleet: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    profiles: dict[int, dict[str, float]] = {}
    for d in fleet:
        rank = int(d["rank"])
        profiles[rank] = {
            "flops": float(d["flops"]),
            "ul_bw": float(d["ul_bw"]),
            "dl_bw": float(d["dl_bw"]),
            "ul_lat": float(d.get("ul_lat", 0.0)),
            "dl_lat": float(d.get("dl_lat", 0.0)),
        }
    return profiles


# -- model config helpers --


def _build_model_config(model_name: str) -> ModelConfig:
    cfg = MODEL_CONFIGS[model_name]
    mc = ModelConfig(
        model_name=model_name,
        seq_length=SEQ_LEN,
        max_seq_len=SEQ_LEN,
        num_layers=cfg["num_layers"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
    )
    setattr(mc, "batch_size", BATCH_SIZE)
    return mc


# -- topology / strategy helpers --


def _best_pp_dp(num_devices: int, num_layers: int) -> tuple:
    best_pp, best_dp = 1, num_devices
    for pp in range(min(num_devices, num_layers), 0, -1):
        if num_devices % pp == 0:
            best_pp, best_dp = pp, num_devices // pp
            break
    return best_pp, best_dp


def _build_scaling_strategy(
    method: str,
    num_devices: int,
    num_layers: int,
):
    """Build strategy with pinned topologies per the PRD.

    - Cleave: centralized scatter-gather with straggler-aware
      partitioning (MIP/greedy sizes (alpha_i, beta_i) blocks
      per device capability; infinite coordinator bw assumed)
    - DTFM:   PP=2 fixed, DP=N/2
    - Asteroid: PP=8 fixed, DP=N/8
    - Confident: re-optimize via _best_pp_dp
    - Alpa: auto-tuned 3D search
    """
    if method == "cleave":
        return CleaveStrategy()

    if method == "dtfm":
        pp, dp = 2, num_devices // 2
        return DTFMStrategy(
            pp_size=pp,
            dp_size=dp,
            population_size=10,
            gcma_trails=50,
            global_batch_size=BATCH_SIZE,
            micro_batch_size=MICRO_BATCH_SIZE,
        )

    if method == "asteroid":
        pp = 8
        return AsteroidStrategy(
            num_stages=pp,
            micro_batch_size=MICRO_BATCH_SIZE,
            num_microbatches=NUM_MICROBATCHES,
        )

    if method == "confident":
        pp, dp = _best_pp_dp(num_devices, num_layers)
        return ConfidentStrategy(pp_size=pp, dp_size=dp)

    if method == "alpa":
        return AlpaStrategy(
            num_microbatches=NUM_MICROBATCHES,
            micro_batch_size=MICRO_BATCH_SIZE,
        )

    raise ValueError(f"Unknown method: {method}")


def _gemm_tuple_dag(
    dag: list[list[dict[str, Any]]],
) -> list[list[tuple[int, int, int]]]:
    return [
        [(int(item["m"]), int(item["n"]), int(item["q"])) for item in level]
        for level in dag
    ]


def _can_run(method: str, num_devices: int) -> bool:
    """Check if method is valid for this device count."""
    if method == "dtfm" and num_devices < 2:
        return False
    if method == "asteroid" and (num_devices < 8 or num_devices % 8):
        return False
    if method == "dtfm" and num_devices % 2:
        return False
    return True


def _run_analytical(
    method: str,
    model_name: str,
    num_devices: int,
    base_fleet: list[dict[str, Any]],
) -> BatchRuntimeResult | None:
    """Run one analytical prediction. Returns None on failure."""
    if not _can_run(method, num_devices):
        return None

    model_config = _build_model_config(model_name)
    fleet = _repeat_fleet(base_fleet, num_devices)
    topology = fleet_to_topology(fleet)
    profiles = _fleet_profiles(fleet)
    gemm_dag = gemm_dag_from_model_config(model_config)

    strategy = _build_scaling_strategy(
        method, num_devices, model_config.num_layers
    )

    class _Timeout(Exception):
        pass

    def _handler(signum, frame):
        raise _Timeout

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(60)
    try:
        if method == "cleave":
            plan = strategy.create_plan(
                model_config,
                topology,
                profiler=None,
                gemm_dag=_gemm_tuple_dag(gemm_dag),
            )
        else:
            plan = strategy.create_plan(model_config, topology, profiler=None)
    except _Timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        LOG.warning(
            "%s/%s/n%d: planner timed out", method, model_name, num_devices
        )
        return None
    except Exception as exc:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        LOG.warning(
            "%s/%s/n%d: planner failed: %s",
            method,
            model_name,
            num_devices,
            exc,
        )
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

    manifest = plan_to_manifest(
        plan=plan,
        gemm_dag=gemm_dag,
        baseline_type=method,
        model_config=model_config,
    )

    pp_size = max(1, len(getattr(plan, "partition_points", []) or []) + 1)
    optimizer_tail = float(getattr(plan, "optimizer_tail_ms", 0.0))

    return compute_batch_runtime(
        vtime_events=[],
        manifest=manifest.to_json(),
        baseline_type=method,
        pp_size=pp_size,
        num_microbatches=NUM_MICROBATCHES,
        optimizer_tail_ms=optimizer_tail,
        device_profiles=profiles,
        bytes_per_element=BYTES_PER_ELEMENT,
    )


# -- extrapolation from existing data points --


def _extrapolate(
    results: list[dict], method: str, model: str, target_n: int
) -> BatchRuntimeResult | None:
    import numpy as np

    pts = [
        (r["num_devices"], r["raw_analytical"])
        for r in results
        if r["method"] == method and r["model"] == model
    ]
    if len(pts) < 3:
        return None

    pts.sort()
    ns = [p[0] for p in pts]
    components = [
        "total_ms",
        "compute_ms",
        "network_ms",
        "allreduce_ms",
        "bubble_ms",
    ]
    projected = {}
    for comp in components:
        vals = [p[1][comp] for p in pts]
        pos_vals = [(n, v) for n, v in zip(ns, vals) if v > 0]
        if len(pos_vals) >= 2:
            log_ns = np.log([x[0] for x in pos_vals])
            log_vs = np.log([x[1] for x in pos_vals])
            a, b = np.polyfit(log_ns, log_vs, 1)
            projected[comp] = float(np.exp(b) * target_n**a)
        else:
            projected[comp] = 0.0

    return BatchRuntimeResult(
        baseline_name=method,
        total_runtime_ms=projected["total_ms"],
        compute_time_ms=projected["compute_ms"],
        network_time_ms=projected["network_ms"],
        pipeline_bubble_ms=projected["bubble_ms"],
        allreduce_time_ms=projected["allreduce_ms"],
        optimizer_tail_ms=0.0,
        per_device_breakdown={},
        per_level_breakdown=[],
    )


# -- fallback: load from existing device_scaling --


def _load_existing_analytical(
    method: str, model_name: str, n_devices: int
) -> BatchRuntimeResult | None:
    path = DEVICE_SCALING_DIR / model_name / f"n{n_devices}" / "summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    baselines = data.get("baselines", {})
    if method not in baselines:
        return None
    b = baselines[method]
    return BatchRuntimeResult(
        baseline_name=method,
        total_runtime_ms=b["total_runtime_ms"],
        compute_time_ms=b["compute_time_ms"],
        network_time_ms=b["network_time_ms"],
        pipeline_bubble_ms=b["pipeline_bubble_ms"],
        allreduce_time_ms=b["allreduce_time_ms"],
        optimizer_tail_ms=b.get("optimizer_tail_ms", 0.0),
        per_device_breakdown={},
        per_level_breakdown=[],
    )


# -- calibration rescaling --


def _straggler_ratio(fleet: list[dict[str, Any]], dim: str) -> float:
    if dim == "dl":
        vals = [d["dl_bw"] for d in fleet]
    else:
        vals = [d["flops"] for d in fleet]
    inv = [1.0 / v for v in vals]
    return max(inv) / (sum(inv) / len(inv))


def _rescale_factors(
    calibration: dict,
    mild_fleet: list[dict[str, Any]],
    target_fleet: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    rescaled = {}
    for method, mcal in calibration["per_method"].items():
        factors = dict(mcal["correction_factors"])
        sens = mcal.get("straggler_sensitivity")
        if sens is None:
            rescaled[method] = factors
            continue
        dim = sens["dimension"]
        exp = sens["exponent"]
        mild_sr = _straggler_ratio(mild_fleet, dim)
        target_sr = _straggler_ratio(target_fleet, dim)
        scale = (target_sr / mild_sr) ** exp
        rescaled[method] = {k: v * scale for k, v in factors.items()}
    return rescaled


def _apply_calibration(
    raw: BatchRuntimeResult,
    factors: dict[str, float],
) -> dict[str, float]:
    return {
        "total_ms": raw.total_runtime_ms * factors["total"],
        "compute_ms": raw.compute_time_ms * factors["compute"],
        "network_ms": raw.network_time_ms * factors["network"],
        "allreduce_ms": raw.allreduce_time_ms * factors["allreduce"],
        "bubble_ms": raw.pipeline_bubble_ms * factors["bubble"],
    }


def _raw_dict(r: BatchRuntimeResult) -> dict[str, float]:
    return {
        "total_ms": r.total_runtime_ms,
        "compute_ms": r.compute_time_ms,
        "network_ms": r.network_time_ms,
        "allreduce_ms": r.allreduce_time_ms,
        "bubble_ms": r.pipeline_bubble_ms,
    }


# -- main projection --


def project(fleet_profile: str | None = None):
    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)

    base_fleet = _load_base_fleet(profile=fleet_profile)
    models = list(MODEL_CONFIGS.keys())

    if fleet_profile is not None:
        mild_fleet = _load_base_fleet(profile=None)
        rescaled = _rescale_factors(calibration, mild_fleet, base_fleet)
        LOG.info("Rescaled correction factors for '%s' profile", fleet_profile)
        for m, f in rescaled.items():
            orig = calibration["per_method"][m]["correction_factors"]["total"]
            LOG.info("  %s: %.2f -> %.2f", m, orig, f["total"])
    else:
        rescaled = None

    # Resume from existing results if available
    results = []
    existing_keys: set[tuple] = set()
    if PROJECTIONS_PATH.exists():
        with open(PROJECTIONS_PATH) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        for r in results:
            existing_keys.add((r["num_devices"], r["model"], r["method"]))
        LOG.info("Resuming with %d existing results", len(results))

    total = len(DEVICE_COUNTS) * len(models) * len(METHODS)
    done = 0

    for n_devices in DEVICE_COUNTS:
        for model_name in models:
            for method in METHODS:
                done += 1
                key = (n_devices, model_name, method)
                if key in existing_keys:
                    continue
                if not _can_run(method, n_devices):
                    LOG.info(
                        "[%d/%d] SKIP %s/%s/n%d",
                        done,
                        total,
                        method,
                        model_name,
                        n_devices,
                    )
                    continue

                LOG.info(
                    "[%d/%d] %s/%s/n%d",
                    done,
                    total,
                    method,
                    model_name,
                    n_devices,
                )

                raw = None
                if fleet_profile is None:
                    raw = _load_existing_analytical(
                        method, model_name, n_devices
                    )
                    if raw is not None:
                        LOG.info("  -> loaded from device_scaling")
                if raw is None and n_devices <= 512:
                    raw = _run_analytical(
                        method, model_name, n_devices, base_fleet
                    )
                    if raw is None:
                        LOG.warning("  -> FAILED")
                        continue
                else:
                    extrap = _extrapolate(
                        results, method, model_name, n_devices
                    )
                    if extrap is not None:
                        raw = extrap
                        LOG.info("  -> extrapolated from trend")
                    else:
                        LOG.info("  -> skip (no data to extrapolate)")
                        continue

                if rescaled is not None:
                    factors = rescaled[method]
                else:
                    factors = calibration["per_method"][method][
                        "correction_factors"
                    ]
                calibrated = _apply_calibration(raw, factors)

                results.append(
                    {
                        "num_devices": n_devices,
                        "model": model_name,
                        "method": method,
                        "raw_analytical": _raw_dict(raw),
                        "calibrated": calibrated,
                    }
                )

                if len(results) % 10 == 0:
                    _save_projections(results, models)

    _save_projections(results, models)
    print(f"Projections: {len(results)} results -> {PROJECTIONS_PATH}")


def _save_projections(results, models):
    output = {
        "sweep_config": {
            "device_counts": DEVICE_COUNTS,
            "models": models,
            "methods": METHODS,
        },
        "results": results,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROJECTIONS_PATH, "w") as f:
        json.dump(output, f, indent=2)


# -- validation (D4) --


def validate():
    with open(PROJECTIONS_PATH) as f:
        projections = json.load(f)

    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)

    comparisons = []
    for entry in projections["results"]:
        model = entry["model"]
        n = entry["num_devices"]
        method = entry["method"]

        existing_path = DEVICE_SCALING_DIR / model / f"n{n}" / "summary.json"
        if not existing_path.exists():
            continue

        with open(existing_path) as f:
            existing = json.load(f)

        baselines = existing.get("baselines", {})
        if method not in baselines:
            continue

        existing_total = baselines[method]["total_runtime_ms"]
        calibrated_total = entry["calibrated"]["total_ms"]

        if existing_total <= 0:
            continue

        rel_error = abs(calibrated_total - existing_total) / existing_total

        if rel_error < 0.5:
            status = "PASS"
        elif rel_error < 2.0:
            status = "WARN"
        else:
            status = "FAIL"

        comparisons.append(
            {
                "model": model,
                "num_devices": n,
                "method": method,
                "calibrated_total_ms": calibrated_total,
                "existing_analytical_total_ms": existing_total,
                "relative_error": round(rel_error, 4),
                "status": status,
            }
        )

    pass_count = sum(1 for c in comparisons if c["status"] == "PASS")
    warn_count = sum(1 for c in comparisons if c["status"] == "WARN")
    fail_count = sum(1 for c in comparisons if c["status"] == "FAIL")

    validation = {
        "comparisons": comparisons,
        "summary": {
            "total_comparisons": len(comparisons),
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "max_relative_error": (
                max(c["relative_error"] for c in comparisons)
                if comparisons
                else 0.0
            ),
        },
    }

    with open(VALIDATION_PATH, "w") as f:
        json.dump(validation, f, indent=2)

    print(f"Validation: {len(comparisons)} comparisons")
    print(f"  PASS={pass_count}  WARN={warn_count}  FAIL={fail_count}")
    if comparisons:
        print(
            f"  Max relative error: "
            f"{validation['summary']['max_relative_error']:.2%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Analytical scaling projections"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Cross-validate against existing device_scaling results",
    )
    parser.add_argument(
        "--fleet-profile",
        default=None,
        choices=["mild", "moderate", "extreme"],
        help="Fleet heterogeneity profile (default: load saved fleet)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    if args.validate:
        validate()
    else:
        project(fleet_profile=args.fleet_profile)


if __name__ == "__main__":
    main()
