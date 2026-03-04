#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


SCS_PRESETS = {
    15: {"slot_ns": 1_000_000, "budget_min_us": 500, "budget_max_us": 714},
    30: {"slot_ns": 500_000, "budget_min_us": 200, "budget_max_us": 429},
    60: {"slot_ns": 250_000, "budget_min_us": 150, "budget_max_us": 411},
    120: {"slot_ns": 125_000, "budget_min_us": 100, "budget_max_us": 321},
}


def _fgn_autocov(H: float, k: int) -> float:
    k_abs = abs(int(k))
    return 0.5 * (
        abs(k_abs - 1) ** (2.0 * H)
        - 2.0 * (k_abs ** (2.0 * H))
        + (k_abs + 1) ** (2.0 * H)
    )


def generate_fgn(
    n: int, H: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    if n <= 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([rng.normal()], dtype=np.float64)

    r = np.array([_fgn_autocov(H, k) for k in range(n)], dtype=np.float64)
    g = np.concatenate([r, [0.0], r[1:][::-1]])
    m = g.size

    eig = np.fft.fft(g).real
    eig = np.maximum(eig, 0.0)

    V = np.zeros(m, dtype=np.complex128)
    V[0] = np.sqrt(eig[0] / m) * rng.normal()
    V[n] = np.sqrt(eig[n] / m) * rng.normal()
    for k in range(1, n):
        a = rng.normal()
        b = rng.normal()
        scale = np.sqrt(eig[k] / (2.0 * m))
        V[k] = scale * (a + 1j * b)
        V[m - k] = np.conj(V[k])

    x = np.fft.fft(V).real[:n]
    return x.astype(np.float64)


def load_to_sm(
    load: NDArray[np.float64] | float,
    sm_min: int,
    sm_max: int,
    sm_step: int,
) -> NDArray[np.int64]:
    arr = np.asarray(load, dtype=np.float64)
    steps = (sm_max - sm_min) / sm_step
    if steps > 0:
        max_load = 1.0 - 0.5 / steps - 1e-9
    else:
        max_load = 1.0
    arr = np.clip(arr, 0.0, max_load)
    target = sm_min + arr * float(sm_max - sm_min)
    q = np.rint((target - sm_min) / sm_step)
    sm = sm_min + q * sm_step
    sm = np.clip(sm, sm_min, sm_max)
    return sm.astype(np.int64)


def generate_random_load(
    n: int,
    H: float = 0.85,
    mean_rho: float = 0.40,
    seed: int | None = None,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    noise = generate_fgn(n, H, rng)
    std = float(np.std(noise))
    if std < 1e-12:
        norm = np.zeros_like(noise)
    else:
        norm = noise / std
    load = mean_rho + 0.18 * norm
    return np.clip(load, 0.0, 1.0)


def generate_shaped_load(
    n: int,
    H: float = 0.85,
    mean_rho: float = 0.40,
    alpha: float = 0.15,
    slew_cap: float = 0.08,
    seed: int | None = None,
) -> NDArray[np.float64]:
    raw = generate_random_load(n=n, H=H, mean_rho=mean_rho, seed=seed)
    if n <= 0:
        return raw
    shaped = np.empty_like(raw)
    state = float(raw[0])
    shaped[0] = state
    deadband = max(0.04, 0.5 * slew_cap)
    for i in range(1, n):
        ewma = alpha * float(raw[i]) + (1.0 - alpha) * state
        delta = np.clip(ewma - state, -slew_cap, slew_cap)
        next_state = float(np.clip(state + delta, 0.0, 1.0))
        if abs(next_state - state) < deadband:
            next_state = state
        state = next_state
        shaped[i] = state
    return shaped


def generate_timing(
    n: int,
    scs: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    preset = SCS_PRESETS[scs]
    slot_ns_nom = int(preset["slot_ns"])
    bmin_us = int(preset["budget_min_us"])
    bmax_us = int(preset["budget_max_us"])

    sched = np.empty(n, dtype=np.int64)
    decode = np.empty(n, dtype=np.int64)

    t = 0
    for i in range(n):
        sched[i] = t
        budget_us = int(rng.integers(bmin_us, bmax_us + 1))
        decode[i] = t + budget_us * 1_000

        jitter = float(rng.uniform(-0.02, 0.02))
        slot_ns = max(1, int(round(slot_ns_nom * (1.0 + jitter))))
        t += slot_ns

    return sched, decode


def _sm_step_from_pynvml() -> int | None:
    try:
        pynvml = importlib.import_module("pynvml")

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count <= 0:
            pynvml.nvmlShutdown()
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        major, _minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        pynvml.nvmlShutdown()
        return 8 if int(major) >= 9 else 2
    except Exception:
        return None


def _sm_step_from_nvidia_smi() -> int | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = proc.stdout.strip().splitlines()[0].strip()
        major = int(line.split(".")[0])
        return 8 if major >= 9 else 2
    except Exception:
        return None


def detect_sm_step() -> int:
    step = _sm_step_from_pynvml()
    if step is not None:
        return step
    step = _sm_step_from_nvidia_smi()
    if step is not None:
        return step
    return 2


def _stability_runs(sm: NDArray[np.int64]) -> list[int]:
    if sm.size == 0:
        return []
    runs: list[int] = []
    run = 1
    for i in range(1, sm.size):
        if sm[i] == sm[i - 1]:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic RAN scheduling traces for bench_trace_switch"
    )
    parser.add_argument(
        "-n",
        "--num-slots",
        type=int,
        default=10_000,
        help="Number of slots to generate",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["random", "shaped"],
        default="random",
        help="Traffic mode",
    )
    parser.add_argument(
        "--scs",
        type=int,
        choices=[15, 30, 60, 120],
        default=30,
        help="Subcarrier spacing preset (kHz)",
    )
    parser.add_argument("--sm-min", type=int, default=8, help="Minimum SM count")
    parser.add_argument("--sm-max", type=int, default=64, help="Maximum SM count")
    parser.add_argument(
        "--sm-step",
        default="auto",
        help="SM quantization step (auto or integer)",
    )
    parser.add_argument("--hurst", type=float, default=0.85, help="Hurst parameter")
    parser.add_argument(
        "--mean-load",
        type=float,
        default=0.40,
        help="Mean normalized load rho",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.15,
        help="EWMA alpha for shaped mode",
    )
    parser.add_argument(
        "--slew",
        type=float,
        default=0.08,
        help="Per-slot slew cap for shaped mode",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "-o",
        "--output",
        default="data/trace_synthetic.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_slots <= 0:
        raise ValueError("--num-slots must be > 0")
    if not (0.0 < args.hurst < 1.0):
        raise ValueError("--hurst must be in (0, 1)")
    if not (0.0 <= args.mean_load <= 1.0):
        raise ValueError("--mean-load must be in [0, 1]")
    if args.sm_min <= 0 or args.sm_max < args.sm_min:
        raise ValueError("Require 0 < --sm-min <= --sm-max")

    if str(args.sm_step).lower() == "auto":
        sm_step = detect_sm_step()
    else:
        sm_step = int(args.sm_step)
    if sm_step <= 0:
        raise ValueError("--sm-step must be positive")

    rng = np.random.default_rng(args.seed)

    if args.mode == "random":
        load = generate_random_load(
            n=args.num_slots,
            H=args.hurst,
            mean_rho=args.mean_load,
            seed=args.seed,
        )
    else:
        load = generate_shaped_load(
            n=args.num_slots,
            H=args.hurst,
            mean_rho=args.mean_load,
            alpha=args.alpha,
            slew_cap=args.slew,
            seed=args.seed,
        )

    sched_ns, decode_ns = generate_timing(args.num_slots, args.scs, rng)
    sm_count = load_to_sm(load, args.sm_min, args.sm_max, sm_step)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["time_slot_sched_ns", "time_decode_start_est_ns", "sm_count"]
        )
        writer.writerows(
            zip(sched_ns.tolist(), decode_ns.tolist(), sm_count.tolist())
        )

    transitions = int(np.count_nonzero(sm_count[1:] != sm_count[:-1]))
    total_edges = max(1, args.num_slots - 1)
    transition_rate = transitions / total_edges
    budgets_us = (decode_ns - sched_ns) / 1_000.0
    unique_sms = np.unique(sm_count)

    print(f"Mode: {args.mode}")
    print(f"SCS: {args.scs} kHz")
    print(
        f"Unique SMs: {unique_sms.size} "
        f"(min={int(unique_sms.min())}, max={int(unique_sms.max())})"
    )
    print(
        f"SM transitions: {transitions}/{total_edges} "
        f"({100.0 * transition_rate:.2f}%)"
    )
    print(
        f"Budget range (us): {budgets_us.min():.0f}..{budgets_us.max():.0f}"
    )

    if args.mode == "shaped":
        runs = _stability_runs(sm_count)
        mean_run = float(np.mean(runs)) if runs else 0.0
        print(
            "Stability runs: "
            f"count={len(runs)}, mean={mean_run:.2f}, max={max(runs) if runs else 0}"
        )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
