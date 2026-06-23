from __future__ import annotations

import time
from typing import Any, Optional, cast

import torch
import torch.cuda

from .base import ROLE_NAMES, StreamBundle


class CppBackend:
    def __init__(
        self,
        gpu_id: int = 0,
        num_partitions: int = 1,
        partition_idx: int = 0,
        roles: Optional[list[str]] = None,
        stream_priority: int = -1,
        strict: bool = False,
        switch_sync: str = "event_chain",
    ):
        from morphling._GreenCtx import (  # pyright: ignore[reportMissingImports]
            create_runtime,
        )

        self._rt: Any = create_runtime(
            gpu_id=gpu_id,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
            roles=roles or list(ROLE_NAMES),
            stream_priority=stream_priority,
            strict=strict,
            switch_sync=switch_sync,
        )
        self._gpu_id: int = gpu_id
        self._closed: bool = False
        self._swap_log: list[dict[str, int]] = []
        self._swap_count: int = 0
        self._total_python_overhead_ns: int = 0
        self._prev_sm_count: int = 0
        self._rt: Any
        self._stream_cache: dict[tuple[int, str], object] = {}
        if self._rt.is_supported():
            for sm in self._rt.available_sm_counts():
                for role in ROLE_NAMES:
                    ptr = self._rt.get_stream_ptr(sm, role)
                    if ptr:
                        stream = torch.cuda.ExternalStream(ptr, device=gpu_id)
                        self._stream_cache[(sm, role)] = stream

    def is_supported(self) -> bool:
        return self._rt.is_supported()

    def unsupported_reason(self) -> str:
        return self._rt.unsupported_reason()

    def available_sm_counts(self) -> list[int]:
        return self._rt.available_sm_counts()

    def sm_step(self) -> int:
        return self._rt.sm_step()

    def partition_sm_count(self) -> int:
        return self._rt.partition_sm_count()

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        gen = self._rt.generation()
        comp_stream = cast(
            torch.cuda.Stream, self._stream_cache[(sm_count, "compute")]
        )
        recv_stream = cast(
            torch.cuda.Stream, self._stream_cache[(sm_count, "recv")]
        )
        send_stream = cast(
            torch.cuda.Stream, self._stream_cache[(sm_count, "send")]
        )
        dp_stream = cast(
            torch.cuda.Stream, self._stream_cache[(sm_count, "dp")]
        )
        return StreamBundle(
            comp=comp_stream,
            recv=recv_stream,
            send=send_stream,
            dp=dp_stream,
            sm_count=sm_count,
            generation=gen,
        )

    def activate_for_step(self, step_or_time: int) -> tuple[int, int]:
        sm = self._rt.sm_count_at_step(step_or_time)
        _ = self._rt.activate_sm_for_thread(sm)
        return sm, self._rt.generation()

    def activate_for_time(self, elapsed_us: int) -> tuple[int, int]:
        t0 = time.perf_counter_ns()
        sm = self._rt.sm_count_at_time(elapsed_us)
        _ = self._rt.activate_sm_for_thread(sm)
        t1 = time.perf_counter_ns()

        if sm != self._prev_sm_count:
            overhead_ns = t1 - t0
            self._swap_log.append(
                {
                    "timestamp_ns": t0,
                    "from_sm": self._prev_sm_count,
                    "to_sm": sm,
                    "python_overhead_ns": overhead_ns,
                }
            )
            self._swap_count += 1
            self._total_python_overhead_ns += overhead_ns
            self._prev_sm_count = sm

        return sm, self._rt.generation()

    def deactivate(self, prev_sm_count: int) -> None:
        t0 = time.perf_counter_ns()
        self._rt.deactivate_for_thread(prev_sm_count)
        t1 = time.perf_counter_ns()

        if prev_sm_count != self._prev_sm_count:
            overhead_ns = t1 - t0
            self._swap_log.append(
                {
                    "timestamp_ns": t0,
                    "from_sm": self._prev_sm_count,
                    "to_sm": prev_sm_count,
                    "python_overhead_ns": overhead_ns,
                }
            )
            self._swap_count += 1
            self._total_python_overhead_ns += overhead_ns
            self._prev_sm_count = prev_sm_count

    def load_trace(self, path: str) -> bool:
        return self._rt.load_trace(path)

    def sm_count_at_time(self, elapsed_us: int) -> int:
        return self._rt.sm_count_at_time(elapsed_us)

    def sm_count_at_step(self, step: int) -> int:
        return self._rt.sm_count_at_step(step)

    def switch_count(self) -> int:
        return self._rt.switch_count()

    def get_swap_stats(self):
        return {
            "count": self._swap_count,
            "total_overhead_us": self._total_python_overhead_ns / 1000,
            "avg_overhead_us": (
                self._total_python_overhead_ns / 1000 / max(self._swap_count, 1)
            ),
        }

    def get_swap_log(self):
        return list(self._swap_log)

    def reset_swap_stats(self):
        self._swap_log.clear()
        self._swap_count = 0
        self._total_python_overhead_ns = 0
        self._prev_sm_count = 0

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream_cache.clear()
            self._rt.close()
