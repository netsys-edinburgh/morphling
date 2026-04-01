#!/usr/bin/env python3
"""Benchmark GFLOPS per SM count for paper figures.

Usage (Docker with GPU):
  python3 scripts/bench_gflops_per_sm.py \
      --gemm-shapes results/gemm_shapes.json \
      --output results/gflops_per_sm.json
"""

import argparse
import json
import os

import numpy as np
import torch

ALLOWED_SM_COUNTS = {8, 16, 24, 48}


def benchmark_gemm(
    m: int,
    n: int,
    k: int,
    n_warmup: int = 10,
    n_iters: int = 100,
    dtype=torch.float16,
) -> dict:
    """Benchmark a single GEMM shape, return timing stats."""
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(n_warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    # Measure using CUDA events for accuracy
    times_ms: list[float] = []
    for _ in range(n_iters):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        torch.matmul(a, b)
        end_evt.record()
        torch.cuda.synchronize()
        times_ms.append(start_evt.elapsed_time(end_evt))

    arr = np.array(times_ms)
    flops = 2.0 * m * n * k
    mean_ms = float(np.mean(arr))
    gflops = flops / (mean_ms * 1e-3) / 1e9 if mean_ms > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_gflops": gflops,
        "std_gflops": float(np.std(flops / (arr * 1e-3) / 1e9))
        if np.all(arr > 0)
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GFLOPS benchmark per SM count"
    )
    parser.add_argument(
        "--gemm-shapes",
        default="results/gemm_shapes.json",
        help="GEMM shapes JSON from Task 1",
    )
    parser.add_argument(
        "--output",
        default="results/gflops_per_sm.json",
    )
    parser.add_argument(
        "--sm-counts",
        default="8,16,24,48",
        help="Comma-separated SM counts",
    )
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--total-sms", type=int, default=48)
    args = parser.parse_args()

    sm_counts = sorted(int(x) for x in args.sm_counts.split(","))
    for s in sm_counts:
        if s not in ALLOWED_SM_COUNTS:
            print(f"WARNING: SM count {s} not in {ALLOWED_SM_COUNTS}")

    with open(args.gemm_shapes) as f:
        shapes = json.load(f)
    print(f"Benchmarking {len(shapes)} GEMM shapes at SM counts: {sm_counts}")

    # Try green context
    has_green_ctx = False
    rt = None
    try:
        from morphling._GreenCtx import GreenContextRuntime

        rt = GreenContextRuntime(0, args.total_sms)
        if rt.is_supported():
            has_green_ctx = True
            print("Green context runtime: supported")
        else:
            print("Green context: NOT supported on this GPU")
    except Exception as e:
        print(f"Green context unavailable: {e}")

    results: dict = {}
    for sm in sm_counts:
        sm_key = str(sm)
        results[sm_key] = {}
        print(f"\n=== SM count: {sm} ===")

        # Activate partition
        prev_sm = 0
        if has_green_ctx and rt and sm < args.total_sms:
            try:
                prev_sm = rt.activate_sm_for_thread(sm)
                print(f"  Green context: {sm} SMs active")
            except Exception as e:
                print(f"  WARNING: activate failed: {e}")

        for shape in shapes:
            m, n, k = shape["m"], shape["n"], shape["k"]
            gk = f"{m}x{n}x{k}_{shape['phase']}"
            r = benchmark_gemm(m, n, k, args.n_warmup, args.n_iters)
            results[sm_key][gk] = r
            print(
                f"  {gk}: {r['mean_gflops']:.1f} GFLOPS ({r['mean_ms']:.3f} ms)"
            )

        # Deactivate
        if has_green_ctx and rt and sm < args.total_sms:
            try:
                rt.deactivate_for_thread(prev_sm)
            except Exception:
                pass

    # Per-step aggregate
    aggregate: dict = {}
    for sm_key in results:
        total_flops = 0.0
        total_time_s = 0.0
        for shape in shapes:
            gk = f"{shape['m']}x{shape['n']}x{shape['k']}_{shape['phase']}"
            if gk in results[sm_key]:
                cnt = shape.get("count_per_step", 1)
                r = results[sm_key][gk]
                total_flops += 2.0 * shape["m"] * shape["n"] * shape["k"] * cnt
                total_time_s += r["mean_ms"] * cnt / 1000.0
        if total_time_s > 0:
            aggregate[sm_key] = {
                "total_gflops": total_flops / total_time_s / 1e9,
                "total_flops_per_step": total_flops,
                "total_gemm_time_ms": total_time_s * 1000,
            }

    output = {
        "per_shape": results,
        "per_step_aggregate": aggregate,
        "metadata": {
            "sm_counts": sm_counts,
            "n_iters": args.n_iters,
            "dtype": "float16",
        },
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Close runtime
    if rt is not None:
        try:
            rt.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
