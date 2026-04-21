#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportMissingTypeArgument=false, reportUnannotatedClassAttribute=false, reportRedeclaration=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Shard:
    shard_id: int
    device_id: int
    size_bytes: float
    compute_s: float


class DeviceState:
    WRITE_ERR_THRESHOLD_BYTES: float = 15.0 * 1024.0 * 1024.0
    RECONNECT_PENALTY_S: float = 3.0

    def __init__(self, device_id: int, bw_bytes_s: float) -> None:
        self.device_id: int = int(device_id)
        self.bw_bytes_s: float = max(1.0, float(bw_bytes_s))
        self.queue_bytes: float = 0.0
        self.last_t: float = 0.0
        self.compute_available_t: float = 0.0
        self.max_queue_bytes: float = 0.0
        self.write_errors: int = 0
        self.cumulative_penalty_s: float = 0.0

    def advance(self, t: float) -> None:
        if t <= self.last_t:
            return
        drained = self.bw_bytes_s * (t - self.last_t)
        self.queue_bytes = max(0.0, self.queue_bytes - drained)
        self.last_t = t

    def commit(
        self,
        t: float,
        size_bytes: float,
        compute_s: float,
        bloat_threshold_bytes: float = 0.0,
    ) -> float:
        self.advance(t)
        self.queue_bytes += max(0.0, float(size_bytes))
        if self.queue_bytes > self.max_queue_bytes:
            self.max_queue_bytes = self.queue_bytes
        if self.queue_bytes > self.WRITE_ERR_THRESHOLD_BYTES:
            self.write_errors += 1
            self.cumulative_penalty_s += self.RECONNECT_PENALTY_S
        effective_bw = self.bw_bytes_s
        if bloat_threshold_bytes > 0.0 and self.queue_bytes > bloat_threshold_bytes:
            ratio = self.queue_bytes / bloat_threshold_bytes
            effective_bw = self.bw_bytes_s / (1.0 + 0.3 * (ratio - 1.0))
        network_finish_t = t + self.cumulative_penalty_s + (self.queue_bytes / max(1.0, effective_bw))
        compute_start_t = max(network_finish_t, self.compute_available_t)
        finish_t = compute_start_t + max(0.0, float(compute_s))
        self.compute_available_t = finish_t
        return finish_t


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark dispatch overhead with and without backpressure"
    )
    parser.add_argument("--num-devices", type=int, default=256)
    parser.add_argument("--num-slow-devices", type=int, default=3)
    parser.add_argument("--num-shards", type=int, default=800)
    parser.add_argument("--min-shard-mb", type=float, default=0.5)
    parser.add_argument("--max-shard-mb", type=float, default=2.0)
    parser.add_argument("--coord-overhead-ms", type=float, default=0.02)
    parser.add_argument("--tick-ms", type=float, default=5.0)
    parser.add_argument("--watermark-low-mb", type=float, default=4.0)
    parser.add_argument("--watermark-high-mb", type=float, default=10.0)
    parser.add_argument("--global-budget-mb", type=float, default=160.0)
    parser.add_argument("--fairness-cap", type=int, default=4)
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


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _build_bandwidths(
    num_devices: int,
    num_slow: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    bws = [rng.uniform(10.0e6, 100.0e6) for _ in range(num_devices)]
    slow_ids = rng.sample(range(num_devices), k=min(max(0, num_slow), num_devices))
    for idx in slow_ids:
        bws[idx] = 2.0e6
    return bws


def _build_shards(
    num_shards: int,
    num_devices: int,
    min_shard_mb: float,
    max_shard_mb: float,
    seed: int,
) -> list[Shard]:
    rng = random.Random(seed)
    shards: list[Shard] = []
    for shard_id in range(num_shards):
        device_id = shard_id % num_devices
        size_bytes = rng.uniform(min_shard_mb, max_shard_mb) * 1024.0 * 1024.0
        compute_s = rng.uniform(0.004, 0.018)
        shards.append(
            Shard(
                shard_id=shard_id,
                device_id=device_id,
                size_bytes=size_bytes,
                compute_s=compute_s,
            )
        )
    return shards


def _group_pending(shards: list[Shard], num_devices: int) -> dict[int, list[Shard]]:
    grouped: dict[int, list[Shard]] = {i: [] for i in range(num_devices)}
    for shard in shards:
        grouped[shard.device_id].append(shard)
    return grouped


def _total_queue_bytes(states: dict[int, DeviceState], t: float) -> float:
    total = 0.0
    for state in states.values():
        state.advance(t)
        total += state.queue_bytes
    return total


def _record_occupancy(
    states: dict[int, DeviceState],
    t: float,
    samples: list[dict[str, float]],
) -> None:
    total = _total_queue_bytes(states, t)
    peak = 0.0
    for state in states.values():
        if state.queue_bytes > peak:
            peak = state.queue_bytes
    samples.append(
        {
            "t_ms": t * 1000.0,
            "total_queue_mb": total / (1024.0 * 1024.0),
            "peak_queue_mb": peak / (1024.0 * 1024.0),
        }
    )


def _simulate_eager(
    shards: list[Shard],
    bandwidths: list[float],
    coord_overhead_s: float,
    tick_s: float,
) -> dict[str, Any]:
    states = {i: DeviceState(i, bandwidths[i]) for i in range(len(bandwidths))}
    commit_delays_ms: list[float] = []
    finish_times: list[float] = []
    occupancy: list[dict[str, float]] = []
    t = 0.0
    _record_occupancy(states, t, occupancy)
    bloat_thresh = 10.0 * 1024.0 * 1024.0
    for shard in shards:
        finish_t = states[shard.device_id].commit(
            t=t,
            size_bytes=shard.size_bytes,
            compute_s=shard.compute_s,
            bloat_threshold_bytes=bloat_thresh,
        )
        finish_times.append(finish_t)
        commit_delays_ms.append(t * 1000.0)
        t += coord_overhead_s
        if tick_s > 0.0 and len(commit_delays_ms) % 20 == 0:
            _record_occupancy(states, t, occupancy)
    dispatch_round_s = t
    if finish_times:
        batch_runtime_s = max(finish_times)
    else:
        batch_runtime_s = 0.0
    per_conn_peaks = {
        str(did): state.max_queue_bytes / (1024.0 * 1024.0)
        for did, state in states.items()
    }
    _record_occupancy(states, batch_runtime_s, occupancy)
    total_write_errors = sum(s.write_errors for s in states.values())
    return {
        "dispatch_round_ms": dispatch_round_s * 1000.0,
        "peak_queue_per_conn_mb": per_conn_peaks,
        "peak_queue_overall_mb": max((float(v) for v in per_conn_peaks.values()), default=0.0),
        "write_errors": total_write_errors,
        "batch_runtime_ms": batch_runtime_s * 1000.0,
        "queue_occupancy_timeline": occupancy,
    }


def _simulate_backpressure(
    shards: list[Shard],
    bandwidths: list[float],
    coord_overhead_s: float,
    tick_s: float,
    low_wm_bytes: float,
    high_wm_bytes: float,
    global_budget_bytes: float,
    fairness_cap: int,
) -> dict[str, Any]:
    num_devices = len(bandwidths)
    states = {i: DeviceState(i, bandwidths[i]) for i in range(num_devices)}
    pending = _group_pending(shards, num_devices)
    commit_delays_ms: list[float] = []
    finish_times: list[float] = []
    occupancy: list[dict[str, float]] = []

    t = 0.0
    committed = 0
    total = len(shards)
    _record_occupancy(states, t, occupancy)

    while committed < total:
        grants: dict[int, int] = {}
        progress = False
        for did in range(num_devices):
            if not pending[did]:
                continue
            state = states[did]
            state.advance(t)
            if grants.get(did, 0) >= max(1, fairness_cap):
                continue
            next_shard = pending[did][0]
            if state.queue_bytes > high_wm_bytes:
                continue
            if (
                state.queue_bytes > low_wm_bytes
                and state.queue_bytes + next_shard.size_bytes > high_wm_bytes
            ):
                continue
            total_queue = _total_queue_bytes(states, t)
            if total_queue + next_shard.size_bytes > global_budget_bytes:
                continue

            shard = pending[did].pop(0)
            finish_t = state.commit(
                t=t,
                size_bytes=shard.size_bytes,
                compute_s=shard.compute_s,
            )
            finish_times.append(finish_t)
            commit_delays_ms.append(t * 1000.0)
            grants[did] = grants.get(did, 0) + 1
            committed += 1
            progress = True
            t += coord_overhead_s

        if progress:
            _record_occupancy(states, t, occupancy)
            continue

        t += max(1e-6, tick_s)
        _record_occupancy(states, t, occupancy)

    dispatch_round_s = t
    batch_runtime_s = max(finish_times) if finish_times else 0.0
    per_conn_peaks = {
        str(did): state.max_queue_bytes / (1024.0 * 1024.0)
        for did, state in states.items()
    }
    _record_occupancy(states, batch_runtime_s, occupancy)
    total_write_errors = sum(s.write_errors for s in states.values())
    return {
        "dispatch_round_ms": dispatch_round_s * 1000.0,
        "peak_queue_per_conn_mb": per_conn_peaks,
        "peak_queue_overall_mb": max((float(v) for v in per_conn_peaks.values()), default=0.0),
        "write_errors": total_write_errors,
        "batch_runtime_ms": batch_runtime_s * 1000.0,
        "queue_occupancy_timeline": occupancy,
    }


def main() -> int:
    args = _parse_args()
    repo_root = _repo_root()
    output_json = (
        args.output_json
        if args.output_json is not None
        else repo_root
        / "results"
        / "microbenchmarks"
        / "dispatch_overhead.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    bandwidths = _build_bandwidths(
        num_devices=int(args.num_devices),
        num_slow=int(args.num_slow_devices),
        seed=int(args.seed),
    )
    shards = _build_shards(
        num_shards=int(args.num_shards),
        num_devices=int(args.num_devices),
        min_shard_mb=float(args.min_shard_mb),
        max_shard_mb=float(args.max_shard_mb),
        seed=int(args.seed) + 1,
    )

    coord_overhead_s = float(args.coord_overhead_ms) / 1000.0
    tick_s = float(args.tick_ms) / 1000.0

    t0 = time.perf_counter()
    eager = _simulate_eager(
        shards=shards,
        bandwidths=bandwidths,
        coord_overhead_s=coord_overhead_s,
        tick_s=tick_s,
    )
    eager_wall_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    backpressure = _simulate_backpressure(
        shards=shards,
        bandwidths=bandwidths,
        coord_overhead_s=coord_overhead_s,
        tick_s=tick_s,
        low_wm_bytes=float(args.watermark_low_mb) * 1024.0 * 1024.0,
        high_wm_bytes=float(args.watermark_high_mb) * 1024.0 * 1024.0,
        global_budget_bytes=float(args.global_budget_mb) * 1024.0 * 1024.0,
        fairness_cap=int(args.fairness_cap),
    )
    backpressure_wall_ms = (time.perf_counter() - t0) * 1000.0

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "script": Path(__file__).name,
            "num_devices": int(args.num_devices),
            "num_slow_devices": int(args.num_slow_devices),
            "num_shards": int(args.num_shards),
            "shard_size_mb_range": [float(args.min_shard_mb), float(args.max_shard_mb)],
            "coord_overhead_ms": float(args.coord_overhead_ms),
            "tick_ms": float(args.tick_ms),
            "watermark_low_mb": float(args.watermark_low_mb),
            "watermark_high_mb": float(args.watermark_high_mb),
            "global_budget_mb": float(args.global_budget_mb),
            "fairness_cap": int(args.fairness_cap),
            "wall_clock_ms": {
                "eager": eager_wall_ms,
                "backpressure": backpressure_wall_ms,
            },
        },
        "modes": {
            "eager": eager,
            "backpressure": backpressure,
        },
        "summary": {
            "dispatch_round_ms": {
                "eager": eager["dispatch_round_ms"],
                "backpressure": backpressure["dispatch_round_ms"],
            },
            "peak_queue_overall_mb": {
                "eager": eager["peak_queue_overall_mb"],
                "backpressure": backpressure["peak_queue_overall_mb"],
            },
            "write_errors": {
                "eager": eager["write_errors"],
                "backpressure": backpressure["write_errors"],
            },
            "batch_runtime_ms": {
                "eager": eager["batch_runtime_ms"],
                "backpressure": backpressure["batch_runtime_ms"],
            },
        },
    }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
