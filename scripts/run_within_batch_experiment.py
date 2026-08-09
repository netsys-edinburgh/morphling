#!/usr/bin/env python3
"""C2: within-batch dynamic-performance experiment (Wave C).

Takes the real 256-device OPT-13B VTIME logs (measured base per-GEMM
durations) for every baseline, overlays a C1 within-batch time-varying trace
(per-device compute-throttle + network-rate drift), and reports per-batch
runtime and recovered useful FLOPs versus the fixed-straggler setup across a
variation-magnitude sweep. Post-processing only; reuses compute_batch_runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compute_batch_runtime import (
    _load_device_profiles,
    _load_json,
    compute_batch_runtime,
    parse_vtime_log,
)
from generate_trace import generate_within_batch_traces

BASELINES = ["cleave", "dtfm", "asteroid", "confident", "alpa"]


def _inputs(root: Path, baseline: str) -> tuple[Path, Path, Path]:
    vt_candidates = [
        root / baseline / "vtime.log",
        root / "vtime" / baseline / "vtime.log",
    ]
    vt = next((p for p in vt_candidates if p.exists()), vt_candidates[0])
    mf_candidates = [
        root / f"{baseline}_results" / "manifests" / f"{baseline}_manifest.json",
        root / "planning" / "manifests" / f"{baseline}_manifest.json",
    ]
    mf = next((p for p in mf_candidates if p.exists()), mf_candidates[0])
    dc_candidates = [
        root / f"{baseline}_results" / "generated_device_fleet.json",
        root / "generated_device_fleet.json",
    ]
    dc = next((p for p in dc_candidates if p.exists()), dc_candidates[0])
    return vt, mf, dc


def _useful_flops(manifest: Any) -> float:
    ents = manifest.get("entries", []) if isinstance(manifest, dict) else manifest
    seen: dict[tuple[int, int], float] = {}
    for e in ents:
        key = (int(e.get("level", 0)), int(e.get("gemm_id", 0)))
        if key not in seen:
            m = float(e.get("m_total") or e.get("alpha") or 0.0)
            n = float(e.get("n") or 0.0)
            q = float(e.get("q_total") or e.get("beta") or 0.0)
            seen[key] = 2.0 * m * n * q
    return float(sum(seen.values()))


def _num_devices(manifest: Any) -> int:
    if isinstance(manifest, dict):
        meta = manifest.get("metadata", {})
        nd = int(meta.get("num_devices", 0) or 0)
        if nd > 0:
            return nd
    ents = manifest.get("entries", []) if isinstance(manifest, dict) else manifest
    return len({int(e.get("device_id", -1)) for e in ents if int(e.get("device_id", -1)) >= 0})


def _idle_fraction(result: Any, num_devices: int) -> float:
    provisioned = num_devices * result.total_runtime_ms
    if provisioned <= 0:
        return 0.0
    busy = sum(v["total_ms"] for v in result.per_device_breakdown.values())
    return max(0.0, 1.0 - busy / provisioned)


def _level_devices(manifest: Any) -> dict[int, list[int]]:
    ents = manifest.get("entries", []) if isinstance(manifest, dict) else manifest
    out: dict[int, set[int]] = {}
    for e in ents:
        lvl = int(e.get("level", 0))
        dev = int(e.get("device_id", -1))
        if dev >= 0:
            out.setdefault(lvl, set()).add(dev)
    return {k: sorted(v) for k, v in out.items()}


def _mean_rate(
    trace: dict[str, Any],
    devices: list[int],
    level: int,
    num_levels: int,
    num_bins: int,
) -> float:
    if not devices:
        return 1.0
    frac = 0.0 if num_levels <= 1 else min(0.999999, level / num_levels)
    b = min(num_bins - 1, int(frac * num_bins))
    tdev = trace.get("devices", {})
    rates: list[float] = []
    for d in devices:
        dd = tdev.get(str(d))
        if not dd:
            rates.append(1.0)
            continue
        r = min(dd["flops_mult"][b], dd["ul_bw_mult"][b], dd["dl_bw_mult"][b])
        rates.append(max(1e-6, float(r)))
    return sum(rates) / len(rates)


def _cleave_reweighted_runtime(
    nominal: Any,
    trace: dict[str, Any],
    level_devices: dict[int, list[int]],
    num_bins: int,
    num_levels: int,
) -> float:
    """CLEAVE work-conserving reweighting: per level, the nominal bottleneck
    time is inflated by 1/mean_rate (work redistributed proportional to current
    device rates), so CLEAVE is limited by the AVERAGE device slowdown rather
    than the slowest participant (paper mechanism, evaluation.tex L402)."""
    total = 0.0
    for lvl in nominal.per_level_breakdown:
        l = int(lvl["level"])
        t = float(lvl["runtime_ms"])
        mr = _mean_rate(trace, level_devices.get(l, []), l, num_levels, num_bins)
        total += t / mr if mr > 0 else t
    return total + float(nominal.optimizer_tail_ms)


def run(
    root: Path,
    magnitudes: list[float],
    num_bins: int,
    event_rate: float,
    duration_frac: float,
    seed: int,
) -> dict[str, Any]:
    cache: dict[str, dict[str, Any]] = {}
    for b in BASELINES:
        vt, mf, dc = _inputs(root, b)
        if not (vt.exists() and mf.exists()):
            print(f"skip {b}: missing inputs ({vt.exists()=}, {mf.exists()=})")
            continue
        manifest = _load_json(str(mf))
        ld = _level_devices(manifest)
        cache[b] = {
            "vtime": parse_vtime_log(str(vt)),
            "manifest": manifest,
            "profiles": _load_device_profiles(str(dc)) if dc.exists() else {},
            "useful_flops": _useful_flops(manifest),
            "num_devices": _num_devices(manifest),
            "level_devices": ld,
            "num_levels": max(ld.keys(), default=0) + 1,
            "nominal": compute_batch_runtime(
                vtime_events=parse_vtime_log(str(vt)),
                manifest=manifest,
                baseline_type=b,
                device_profiles=_load_device_profiles(str(dc)) if dc.exists() else {},
            ),
        }

    per_baseline: dict[str, dict[str, Any]] = {b: {"points": []} for b in cache}
    base_runtime: dict[str, float] = {
        b: cache[b]["nominal"].total_runtime_ms for b in cache
    }

    for m in magnitudes:
        nd = max((cache[b]["num_devices"] for b in cache), default=256)
        trace = (
            None
            if m <= 0.0
            else generate_within_batch_traces(
                num_devices=nd,
                num_bins=num_bins,
                magnitude=m,
                event_rate=event_rate,
                duration_frac=duration_frac,
                network_drift_std=0.5 * m,
                hurst=0.85,
                seed=seed,
            )
        )
        for b, data in cache.items():
            idle: float | None
            if m <= 0.0:
                rt = base_runtime[b]
                idle = _idle_fraction(data["nominal"], data["num_devices"])
            elif b == "cleave":
                rt = _cleave_reweighted_runtime(
                    data["nominal"], trace, data["level_devices"],
                    num_bins, data["num_levels"],
                )
                idle = None
            else:
                result = compute_batch_runtime(
                    vtime_events=data["vtime"],
                    manifest=data["manifest"],
                    baseline_type=b,
                    device_profiles=data["profiles"],
                    within_batch_trace=trace,
                )
                rt = result.total_runtime_ms
                idle = _idle_fraction(result, data["num_devices"])
            per_baseline[b]["points"].append(
                {
                    "magnitude": m,
                    "runtime_ms": rt,
                    "normalized_runtime": rt / base_runtime[b]
                    if base_runtime.get(b)
                    else 1.0,
                    "idle_fraction": idle,
                    "useful_flop_throughput": data["useful_flops"] / rt
                    if rt > 0
                    else 0.0,
                }
            )

    recovered = _recovered_useful_flops(cache, per_baseline, base_runtime, magnitudes)

    return {
        "experiment": "within_batch_dynamic_performance",
        "model": "opt-13b",
        "num_devices": max((cache[b]["num_devices"] for b in cache), default=0),
        "magnitudes": magnitudes,
        "variation_model": {
            "num_bins": num_bins,
            "event_rate": event_rate,
            "duration_frac": duration_frac,
            "network_drift_std": "0.5 * magnitude",
            "seed": seed,
        },
        "base_runtime_ms": base_runtime,
        "per_baseline": per_baseline,
        "recovered_useful_flops": recovered,
    }


def _recovered_useful_flops(
    cache: dict[str, Any],
    per_baseline: dict[str, Any],
    base_runtime: dict[str, float],
    magnitudes: list[float],
) -> dict[str, Any]:
    if "cleave" not in cache:
        return {}
    baselines = [b for b in per_baseline if b != "cleave"]
    out: list[dict[str, Any]] = []
    for i, m in enumerate(magnitudes):
        if m <= 0.0:
            continue
        cl = per_baseline["cleave"]["points"][i]
        useful = cache["cleave"]["useful_flops"]
        cl_lost = useful * max(
            0.0, 1.0 - base_runtime["cleave"] / cl["runtime_ms"]
        )
        base_losses = []
        for b in baselines:
            pt = per_baseline[b]["points"][i]
            base_losses.append(
                cache[b]["useful_flops"]
                * max(0.0, 1.0 - base_runtime[b] / pt["runtime_ms"])
            )
        mean_base_lost = sum(base_losses) / len(base_losses) if base_losses else 0.0
        recovered = max(0.0, mean_base_lost - cl_lost)
        out.append(
            {
                "magnitude": m,
                "cleave_lost_flops": cl_lost,
                "mean_baseline_lost_flops": mean_base_lost,
                "recovered_flops": recovered,
                "recovered_fraction_of_useful": recovered / useful if useful else 0.0,
            }
        )
    return {"per_magnitude": out}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="results/vtime_models/opt-13b",
        help="Directory with {baseline}/vtime.log and {baseline}_results/",
    )
    parser.add_argument(
        "--magnitudes",
        type=str,
        default="0.0,0.2,0.4,0.6",
        help="Comma-separated variation magnitudes (throttle depth)",
    )
    parser.add_argument("--num-bins", type=int, default=100)
    parser.add_argument("--event-rate", type=float, default=3.0)
    parser.add_argument("--duration-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="results/within_batch/summary.json",
    )
    args = parser.parse_args()

    magnitudes = [float(x) for x in args.magnitudes.split(",")]
    summary = run(
        root=Path(args.root),
        magnitudes=magnitudes,
        num_bins=args.num_bins,
        event_rate=args.event_rate,
        duration_frac=args.duration_frac,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote: {out_path}")
    for b, data in summary["per_baseline"].items():
        pts = data["points"]
        norm = ", ".join(f"{p['normalized_runtime']:.2f}" for p in pts)
        print(f"  {b:10s} normalized runtime: [{norm}]")
    rec = summary.get("recovered_useful_flops", {}).get("per_magnitude", [])
    for r in rec:
        print(
            f"  m={r['magnitude']}: CLEAVE recovers "
            f"{100 * r['recovered_fraction_of_useful']:.1f}% of useful FLOPs "
            f"vs mean baseline"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
