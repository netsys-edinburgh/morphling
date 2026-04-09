from __future__ import annotations

import importlib
import logging
from typing import Optional, Protocol, cast

import torch
import torch.cuda

from .base import ROLE_NAMES, StreamBundle
from .trace import parse_trace_file, sm_count_at_step, sm_count_at_time

logger = logging.getLogger(__name__)


class _CudaProps(Protocol):
    multi_processor_count: int
    major: int
    minor: int


class _GreenContextInstance(Protocol):
    def set_context(self) -> None: ...

    def pop_context(self) -> None: ...


class _GreenContextFactory(Protocol):
    @staticmethod
    def create(num_sms: int, device_id: int) -> _GreenContextInstance: ...


class TorchNativeBackend:
    def __init__(
        self,
        gpu_id: int = 0,
        strict: bool = False,
        **kwargs: object,
    ):
        _ = kwargs
        self._supported: bool = False
        self._reason: str = ""
        self._gpu_id: int = gpu_id
        self._strict: bool = strict
        self._closed: bool = False
        self._switch_count: int = 0
        self._generation: int = 0
        self._active_gc: Optional[_GreenContextInstance] = None
        self._active_sm: int = 0
        self._contexts: dict[int, _GreenContextInstance] = {}
        self._stream_cache: dict[tuple[int, str], object] = {}
        self._sm_counts: list[int] = []
        self._sm_step_val: int = 0
        self._partition_sm: int = 0
        self._trace_entries: list[tuple[int, int]] = []
        self._trace_time_unit: str = "us"
        self._trace_clock_mode: str = "step"

        try:
            mod = importlib.import_module("torch.cuda.green_contexts")
        except ModuleNotFoundError:
            mod = None

        if mod is None:
            self._reason = (
                "torch.cuda.green_contexts not available "
                "(requires PyTorch >= 2.10)"
            )
            if strict:
                raise RuntimeError(self._reason)
            return

        supported = bool(getattr(mod, "SUPPORTED", False))
        green_context = getattr(mod, "GreenContext", None)
        if green_context is None:
            self._reason = (
                "torch.cuda.green_contexts not available "
                "(requires PyTorch >= 2.10)"
            )
            if strict:
                raise RuntimeError(self._reason)
            return

        green_context_factory = cast(_GreenContextFactory, green_context)

        if not supported:
            self._reason = (
                "torch.cuda.green_contexts.SUPPORTED is "
                "False (requires CUDA 12.8+ build)"
            )
            if strict:
                raise RuntimeError(self._reason)
            return

        torch.cuda.init()
        torch.cuda.set_device(gpu_id)

        props = cast(_CudaProps, torch.cuda.get_device_properties(gpu_id))
        total_sms = props.multi_processor_count
        cc = props.major * 10 + props.minor
        self._sm_step_val = 8 if cc >= 90 else 2

        self._sm_counts = list(
            range(
                self._sm_step_val,
                total_sms + 1,
                self._sm_step_val,
            )
        )
        self._partition_sm = self._sm_counts[-1] if self._sm_counts else 0

        created_any = False
        for sm in self._sm_counts:
            try:
                gc = green_context_factory.create(num_sms=sm, device_id=gpu_id)
                self._contexts[sm] = gc
                gc.set_context()
                try:
                    for role in ROLE_NAMES:
                        stream = torch.cuda.Stream(device=gpu_id)
                        self._stream_cache[(sm, role)] = stream
                finally:
                    gc.pop_context()
                created_any = True
            except RuntimeError as e:
                if strict:
                    raise
                logger.warning(
                    "TorchNative: failed to create green ctx with %d SMs: %s",
                    sm,
                    e,
                )
                continue

        if created_any:
            self._supported = True
            self._sm_counts = sorted(self._contexts.keys())
            self._partition_sm = self._sm_counts[-1]
        else:
            self._reason = "No green contexts could be created"

    def is_supported(self) -> bool:
        return self._supported

    def unsupported_reason(self) -> str:
        return self._reason

    def available_sm_counts(self) -> list[int]:
        return list(self._sm_counts)

    def sm_step(self) -> int:
        return self._sm_step_val

    def partition_sm_count(self) -> int:
        return self._partition_sm

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        return StreamBundle(
            comp=self._stream_cache[(sm_count, "compute")],
            recv=self._stream_cache[(sm_count, "recv")],
            send=self._stream_cache[(sm_count, "send")],
            dp=self._stream_cache[(sm_count, "dp")],
            sm_count=sm_count,
            generation=self._generation,
        )

    def activate_for_step(self, step_or_time: int) -> tuple[int, int]:
        if self._trace_entries:
            sm = sm_count_at_step(
                self._trace_entries,
                step_or_time,
                self._partition_sm,
            )
            sm = min(
                self._sm_counts,
                key=lambda x: abs(x - sm),
            )
        else:
            sm = self._partition_sm

        if self._active_gc is not None and self._active_sm != sm:
            self._active_gc.pop_context()
            self._active_gc = None
            self._switch_count += 1
            self._generation += 1

        gc = self._contexts.get(sm)
        if gc is not None and self._active_gc is None:
            gc.set_context()
            self._active_gc = gc
            self._active_sm = sm
            if self._switch_count == 0:
                self._switch_count = 1
                self._generation = 1

        return sm, self._generation

    def activate_for_time(self, elapsed_us: int) -> tuple[int, int]:
        if self._trace_entries:
            sm = sm_count_at_time(
                self._trace_entries,
                elapsed_us,
                self._trace_time_unit,
                self._partition_sm,
            )
            sm = min(
                self._sm_counts,
                key=lambda x: abs(x - sm),
            )
        else:
            sm = self._partition_sm

        if self._active_gc is not None and self._active_sm != sm:
            self._active_gc.pop_context()
            self._active_gc = None
            self._switch_count += 1
            self._generation += 1

        gc = self._contexts.get(sm)
        if gc is not None and self._active_gc is None:
            gc.set_context()
            self._active_gc = gc
            self._active_sm = sm
            if self._switch_count == 0:
                self._switch_count = 1
                self._generation = 1

        return sm, self._generation

    def deactivate(self, prev_sm_count: int) -> None:
        _ = prev_sm_count
        if self._active_gc is not None:
            self._active_gc.pop_context()
            self._active_gc = None

    def load_trace(self, path: str) -> bool:
        try:
            entries, tu, cm = parse_trace_file(path)
            self._trace_entries = entries
            self._trace_time_unit = tu
            self._trace_clock_mode = cm
            return len(entries) > 0
        except Exception as e:
            logger.warning(
                "TorchNative: trace load failed: %s",
                e,
            )
            return False

    def sm_count_at_time(self, elapsed_us: int) -> int:
        if not self._trace_entries:
            return self._partition_sm
        return sm_count_at_time(
            self._trace_entries,
            elapsed_us,
            self._trace_time_unit,
            self._partition_sm,
        )

    def sm_count_at_step(self, step: int) -> int:
        if not self._trace_entries:
            return self._partition_sm
        return sm_count_at_step(
            self._trace_entries,
            step,
            self._partition_sm,
        )

    def switch_count(self) -> int:
        return self._switch_count

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._active_gc is not None:
                try:
                    self._active_gc.pop_context()
                except Exception:
                    pass
                self._active_gc = None
            self._stream_cache.clear()
            self._contexts.clear()
