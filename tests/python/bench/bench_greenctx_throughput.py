#!/usr/bin/env python3
"""Benchmark matmul throughput under green context SM partitions."""

from __future__ import annotations

import importlib
import importlib.util
import os
import statistics
import sys
import types
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


class _CudaStream(Protocol):
    def synchronize(self) -> None: ...


class _CudaEvent(Protocol):
    def record(self, stream: _CudaStream) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, end_event: "_CudaEvent") -> float: ...


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def default_stream(self) -> _CudaStream: ...

    def Event(self, *, enable_timing: bool) -> _CudaEvent: ...

    def stream(self, stream: _CudaStream) -> AbstractContextManager[object]: ...

    def set_device(self, device: int) -> None: ...

    def synchronize(self) -> None: ...


class _TorchApi(Protocol):
    float16: object
    cuda: _CudaApi

    def matmul(self, a: object, b: object) -> object: ...

    def randn(
        self,
        size: tuple[int, int],
        *,
        device: object,
        dtype: object,
    ) -> object: ...

    def device(self, device: str) -> object: ...


class _GreenRuntime(Protocol):
    def activate_sm_for_thread(self, sm_count: int) -> int: ...


torch_mod = cast(object, importlib.import_module("torch"))
torch = cast(_TorchApi, torch_mod)

TORCH_LIB_PATH = (
    "/mnt/data/xly/.conda/envs/emulator/lib/python3.9/site-packages/torch/lib"
)
CUDA_LIB_PATH = "/usr/local/cuda-12.6/lib64"

GPU_PREFERENCE = (7, 6, 5, 0)
SM_COUNTS = (8, 16, 32, 48, 64)
MATRIX_SIZES = (512, 1024, 2048, 4096)
WARMUP_ITERS = 5
BENCH_ITERS = {512: 24, 1024: 16, 2048: 10, 4096: 6}
SWITCH_ITERS = 48
DTYPE = torch.float16


def _ensure_ld_library_path() -> None:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = [p for p in current.split(":") if p]

    for required in (TORCH_LIB_PATH, CUDA_LIB_PATH):
        if required not in entries:
            entries.insert(0, required)

    os.environ["LD_LIBRARY_PATH"] = ":".join(entries)


def _bootstrap_morphling() -> None:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))

    if "morphling" not in sys.modules:
        morphling_mod = types.ModuleType("morphling")
        morphling_mod.__path__ = [str(root / "morphling")]
        morphling_mod.__package__ = "morphling"
        sys.modules["morphling"] = morphling_mod

    if "morphling.runtime" not in sys.modules:
        runtime_mod = types.ModuleType("morphling.runtime")
        runtime_mod.__path__ = [str(root / "morphling" / "runtime")]
        runtime_mod.__package__ = "morphling.runtime"
        sys.modules["morphling.runtime"] = runtime_mod

    so_path = root / "morphling" / "_GreenCtx.so"
    if not so_path.exists():
        raise FileNotFoundError(f"Missing green context runtime: {so_path}")

    if "morphling._GreenCtx" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "morphling._GreenCtx", str(so_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load spec for: {so_path}")
        greenctx_mod = importlib.util.module_from_spec(spec)
        sys.modules["morphling._GreenCtx"] = greenctx_mod
        spec.loader.exec_module(greenctx_mod)


_ensure_ld_library_path()
_bootstrap_morphling()

from morphling.runtime.green_context_backends import CppBackend


@dataclass
class BenchRow:
    matrix_size: int
    sm_count: str
    avg_time_ms: float
    gflops: float
    switch_overhead_us: float


def _pick_gpu_id() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    count = torch.cuda.device_count()
    for gpu_id in GPU_PREFERENCE:
        if gpu_id < count:
            return gpu_id
    return 0


def _calc_gflops(size: int, avg_ms: float) -> float:
    if avg_ms <= 0:
        return 0.0
    ops = 2.0 * size * size * size
    return ops / (avg_ms * 1e6)


def _measure_baseline_ms(
    a: object,
    b: object,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    stream = torch.cuda.default_stream()

    for _ in range(warmup_iters):
        _ = torch.matmul(a, b)
    stream.synchronize()

    times_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)
        _ = torch.matmul(a, b)
        end.record(stream)

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.fmean(times_ms)


def _measure_switch_overhead_us(
    backend: CppBackend,
    sm_count: int,
    iterations: int,
) -> float:
    stream = torch.cuda.default_stream()
    runtime = cast(_GreenRuntime, cast(object, getattr(backend, "_rt")))
    overhead_us: list[float] = []

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)
        prev_sm = runtime.activate_sm_for_thread(sm_count)
        end.record(stream)

        end.synchronize()
        backend.deactivate(prev_sm)
        overhead_us.append(start.elapsed_time(end) * 1000.0)

    return statistics.fmean(overhead_us)


def _measure_greenctx_ms(
    backend: CppBackend,
    a: object,
    b: object,
    sm_count: int,
    warmup_iters: int,
    bench_iters: int,
) -> float:
    runtime = cast(_GreenRuntime, cast(object, getattr(backend, "_rt")))
    bundle = backend.get_stream_bundle(sm_count)
    comp_stream = cast(_CudaStream, bundle.comp)

    for _ in range(warmup_iters):
        prev_sm = runtime.activate_sm_for_thread(sm_count)
        try:
            with torch.cuda.stream(comp_stream):
                _ = torch.matmul(a, b)
            comp_stream.synchronize()
        finally:
            backend.deactivate(prev_sm)

    times_ms: list[float] = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        prev_sm = runtime.activate_sm_for_thread(sm_count)
        try:
            with torch.cuda.stream(comp_stream):
                start.record(comp_stream)
                _ = torch.matmul(a, b)
                end.record(comp_stream)
            end.synchronize()
        finally:
            backend.deactivate(prev_sm)

        times_ms.append(start.elapsed_time(end))

    return statistics.fmean(times_ms)


def _print_table(rows: list[BenchRow]) -> None:
    headers = [
        "Matrix Size",
        "SM Count",
        "Avg Time (ms)",
        "GFLOPS",
        "Switch Overhead (us)",
    ]

    print("| " + " | ".join(headers) + " |")
    print("|---:|:---|---:|---:|---:|")

    for row in rows:
        line = (
            f"| {row.matrix_size} | {row.sm_count} | "
            + f"{row.avg_time_ms:.3f} | {row.gflops:.2f} | "
            + f"{row.switch_overhead_us:.2f} |"
        )
        print(line)


def main() -> None:
    gpu_id = _pick_gpu_id()
    torch.cuda.set_device(gpu_id)

    backend = CppBackend(gpu_id=gpu_id)
    if not backend.is_supported():
        reason = backend.unsupported_reason()
        backend.close()
        raise RuntimeError(
            f"Green contexts not supported on GPU {gpu_id}: {reason}"
        )

    available_sms = set(backend.available_sm_counts())
    missing_sms = [sm for sm in SM_COUNTS if sm not in available_sms]
    if missing_sms:
        backend.close()
        raise RuntimeError(
            f"Missing required SM partitions: {missing_sms}; "
            + f"available={sorted(available_sms)}"
        )

    cfg_line = (
        f"GPU={gpu_id}, dtype={DTYPE}, warmup={WARMUP_ITERS}, "
        + f"sm_counts={list(SM_COUNTS)}"
    )
    print(cfg_line)

    rows: list[BenchRow] = []
    device = torch.device(f"cuda:{gpu_id}")

    try:
        for size in MATRIX_SIZES:
            bench_iters = BENCH_ITERS[size]

            a = torch.randn((size, size), device=device, dtype=DTYPE)
            b = torch.randn((size, size), device=device, dtype=DTYPE)

            baseline_ms = _measure_baseline_ms(
                a,
                b,
                warmup_iters=WARMUP_ITERS,
                bench_iters=bench_iters,
            )
            rows.append(
                BenchRow(
                    matrix_size=size,
                    sm_count="baseline",
                    avg_time_ms=baseline_ms,
                    gflops=_calc_gflops(size, baseline_ms),
                    switch_overhead_us=0.0,
                )
            )

            for sm_count in SM_COUNTS:
                switch_us = _measure_switch_overhead_us(
                    backend,
                    sm_count,
                    iterations=SWITCH_ITERS,
                )
                avg_ms = _measure_greenctx_ms(
                    backend,
                    a,
                    b,
                    sm_count,
                    warmup_iters=WARMUP_ITERS,
                    bench_iters=bench_iters,
                )

                rows.append(
                    BenchRow(
                        matrix_size=size,
                        sm_count=str(sm_count),
                        avg_time_ms=avg_ms,
                        gflops=_calc_gflops(size, avg_ms),
                        switch_overhead_us=switch_us,
                    )
                )

            del a
            del b
            torch.cuda.synchronize()

    finally:
        backend.close()

    _print_table(rows)


if __name__ == "__main__":
    main()
