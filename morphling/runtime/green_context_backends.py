"""Green context backend implementations.

Three backends:
  - CppBackend: wraps morphling._GreenCtx C++ runtime
  - TorchNativeBackend: uses torch.cuda.green_contexts
    (PyTorch >= 2.10, CUDA >= 12.8 build)
  - OffBackend: no-op fallback using regular CUDA streams

Backend selection:
  auto → cpp (if available) → torch_native → off
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
import torch.cuda

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────

ROLE_NAMES = ("compute", "recv", "send", "dp")


@dataclass
class StreamBundle:
    """Pre-cached streams for one SM partition."""

    comp: torch.cuda.Stream
    recv: torch.cuda.Stream
    send: torch.cuda.Stream
    dp: torch.cuda.Stream
    sm_count: int
    generation: int

    def streams_dict(self) -> Dict[str, torch.cuda.Stream]:
        return {
            "compute": self.comp,
            "recv": self.recv,
            "send": self.send,
            "dp": self.dp,
        }


# ── Backend protocol ─────────────────────────────────────


class GreenCtxBackend(Protocol):
    """Interface every backend must implement."""

    def is_supported(self) -> bool:
        ...

    def unsupported_reason(self) -> str:
        ...

    def available_sm_counts(self) -> List[int]:
        ...

    def sm_step(self) -> int:
        ...

    def partition_sm_count(self) -> int:
        ...

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        ...

    def activate_for_step(
        self, step_or_time: int
    ) -> Tuple[int, int]:
        """Returns (sm_count, generation)."""
        ...

    def activate_for_time(
        self, elapsed_us: int
    ) -> Tuple[int, int]:
        """Returns (sm_count, generation)."""
        ...

    def deactivate(self, prev_sm_count: int) -> None:
        ...

    def load_trace(self, path: str) -> bool:
        ...

    def sm_count_at_time(self, elapsed_us: int) -> int:
        ...

    def sm_count_at_step(self, step: int) -> int:
        ...

    def switch_count(self) -> int:
        ...

    def close(self) -> None:
        ...


# ── CppBackend ────────────────────────────────────────────


class CppBackend:
    """Wraps the C++ GreenContextRuntime via pybind11."""

    def __init__(
        self,
        gpu_id: int = 0,
        num_partitions: int = 1,
        partition_idx: int = 0,
        roles: Optional[List[str]] = None,
        stream_priority: int = -1,
        strict: bool = False,
        switch_sync: str = "event_chain",
    ):
        from morphling._GreenCtx import (  # pyright: ignore[reportMissingImports]
            create_runtime,
        )

        self._rt = create_runtime(
            gpu_id=gpu_id,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
            roles=roles or list(ROLE_NAMES),
            stream_priority=stream_priority,
            strict=strict,
            switch_sync=switch_sync,
        )
        self._gpu_id = gpu_id
        self._closed = False

        # Pre-build ExternalStream cache (one-time)
        self._stream_cache: Dict[
            Tuple[int, str], torch.cuda.Stream
        ] = {}
        if self._rt.is_supported():
            for sm in self._rt.available_sm_counts():
                for role in ROLE_NAMES:
                    ptr = self._rt.get_stream_ptr(sm, role)
                    if ptr:
                        stream = torch.cuda.ExternalStream(
                            ptr, device=gpu_id
                        )
                        self._stream_cache[(sm, role)] = stream

    def is_supported(self) -> bool:
        return self._rt.is_supported()

    def unsupported_reason(self) -> str:
        return self._rt.unsupported_reason()

    def available_sm_counts(self) -> List[int]:
        return self._rt.available_sm_counts()

    def sm_step(self) -> int:
        return self._rt.sm_step()

    def partition_sm_count(self) -> int:
        return self._rt.partition_sm_count()

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        gen = self._rt.generation()
        return StreamBundle(
            comp=self._stream_cache[(sm_count, "compute")],
            recv=self._stream_cache[(sm_count, "recv")],
            send=self._stream_cache[(sm_count, "send")],
            dp=self._stream_cache[(sm_count, "dp")],
            sm_count=sm_count,
            generation=gen,
        )

    def activate_for_step(
        self, step_or_time: int
    ) -> Tuple[int, int]:
        sm = self._rt.sm_count_at_step(step_or_time)
        prev = self._rt.activate_sm_for_thread(sm)
        return sm, self._rt.generation()

    def activate_for_time(
        self, elapsed_us: int
    ) -> Tuple[int, int]:
        sm = self._rt.sm_count_at_time(elapsed_us)
        prev = self._rt.activate_sm_for_thread(sm)
        return sm, self._rt.generation()

    def deactivate(self, prev_sm_count: int) -> None:
        self._rt.deactivate_for_thread(prev_sm_count)

    def load_trace(self, path: str) -> bool:
        return self._rt.load_trace(path)

    def sm_count_at_time(self, elapsed_us: int) -> int:
        return self._rt.sm_count_at_time(elapsed_us)

    def sm_count_at_step(self, step: int) -> int:
        return self._rt.sm_count_at_step(step)

    def switch_count(self) -> int:
        return self._rt.switch_count()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream_cache.clear()
            self._rt.close()


# ── Python trace parser (shared by TorchNativeBackend) ──


def _parse_trace_file(
    path: str,
) -> Tuple[
    List[Tuple[int, int]], str, str
]:
    """Parse a green context trace file.

    Returns (entries, time_unit, clock_mode) where entries
    is a list of (timestamp, num_sms) tuples.
    """
    entries: List[Tuple[int, int]] = []
    time_unit = "us"
    clock_mode = "step"
    with open(path) as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle comment-directives: # key=value
            if line.startswith("#"):
                directive = line.lstrip("#").strip()
                if directive.startswith("time_unit="):
                    time_unit = directive.split(
                        "=", 1
                    )[1]
                elif directive.startswith(
                    "clock_mode="
                ):
                    clock_mode = directive.split(
                        "=", 1
                    )[1]
                continue
            # Also handle bare directives (no #)
            if line.startswith("time_unit="):
                time_unit = line.split("=", 1)[1]
                continue
            if line.startswith("clock_mode="):
                clock_mode = line.split("=", 1)[1]
                continue
            # Skip header row (first non-directive
            # non-comment line if non-numeric)
            parts = line.split(",")
            if not header_skipped:
                try:
                    int(parts[0])
                except ValueError:
                    header_skipped = True
                    continue
                header_skipped = True
            if len(parts) >= 2:
                try:
                    ts = int(parts[0])
                    sms = int(parts[1])
                    entries.append((ts, sms))
                except ValueError:
                    continue
    return entries, time_unit, clock_mode


def _sm_count_at_step(
    entries: List[Tuple[int, int]],
    step: int,
    default_sm: int,
) -> int:
    """Step-function lookup: return SM count at given step.

    Entries sorted by timestamp. Returns SM count of the
    last entry whose timestamp <= step. If no entry matches,
    returns default_sm.
    """
    result = default_sm
    for ts, sms in entries:
        if ts <= step:
            result = sms
        else:
            break
    return result


def _sm_count_at_time(
    entries: List[Tuple[int, int]],
    elapsed_us: int,
    time_unit: str,
    default_sm: int,
) -> int:
    """Wall-clock lookup: convert elapsed_us to trace time
    unit and return SM count.
    """
    if time_unit == "s":
        # Entries in seconds, elapsed in microseconds
        scale = 1_000_000
    elif time_unit == "ms":
        scale = 1_000
    else:  # us
        scale = 1
    result = default_sm
    for ts, sms in entries:
        threshold_us = ts * scale
        if elapsed_us >= threshold_us:
            result = sms
        else:
            break
    return result


# ── TorchNativeBackend ────────────────────────────────────


class TorchNativeBackend:
    """Uses torch.cuda.green_contexts (PyTorch >= 2.10,
    CUDA >= 12.8 build).

    Creates one GreenContext per SM count. Streams are
    created via torch.cuda.Stream() while the green
    context is pushed (set_context), binding them to the
    SM-partitioned context. Activation uses
    set_context()/pop_context() for automatic PyTorch
    stream management.

    Requires:
      - PyTorch >= 2.10 with CUDA 12.8+ build
      - torch.cuda.green_contexts.SUPPORTED == True
    """

    def __init__(
        self,
        gpu_id: int = 0,
        strict: bool = False,
        **kwargs,
    ):
        self._supported = False
        self._reason = ""
        self._gpu_id = gpu_id
        self._strict = strict
        self._closed = False
        self._switch_count = 0
        self._generation = 0
        self._active_gc: Optional[Any] = None
        self._active_sm = 0
        self._contexts: Dict[int, Any] = {}
        self._stream_cache: Dict[
            Tuple[int, str], torch.cuda.Stream
        ] = {}
        self._sm_counts: List[int] = []
        self._sm_step_val = 0
        self._partition_sm = 0
        self._trace_entries: List[Tuple[int, int]] = []
        self._trace_time_unit = "us"
        self._trace_clock_mode = "step"

        # Check availability
        try:
            from torch.cuda.green_contexts import (
                SUPPORTED,
                GreenContext,
            )
        except ImportError:
            self._reason = (
                "torch.cuda.green_contexts not available "
                "(requires PyTorch >= 2.10)"
            )
            if strict:
                raise RuntimeError(self._reason)
            return

        if not SUPPORTED:
            self._reason = (
                "torch.cuda.green_contexts.SUPPORTED is "
                "False (requires CUDA 12.8+ build)"
            )
            if strict:
                raise RuntimeError(self._reason)
            return

        # Ensure CUDA initialized
        torch.cuda.init()
        torch.cuda.set_device(gpu_id)

        # Determine SM properties
        props = torch.cuda.get_device_properties(gpu_id)
        total_sms = props.multi_processor_count
        cc = props.major * 10 + props.minor
        self._sm_step_val = 8 if cc >= 90 else 2

        # Available SM counts
        self._sm_counts = list(range(
            self._sm_step_val,
            total_sms + 1,
            self._sm_step_val,
        ))
        self._partition_sm = (
            self._sm_counts[-1] if self._sm_counts else 0
        )

        # Create GreenContext + 4 streams per SM count.
        # Streams are created while the green context is
        # pushed via set_context(), binding them to that
        # SM partition.
        created_any = False
        for sm in self._sm_counts:
            try:
                gc = GreenContext.create(
                    num_sms=sm, device_id=gpu_id
                )
                self._contexts[sm] = gc
                gc.set_context()
                try:
                    for role in ROLE_NAMES:
                        stream = torch.cuda.Stream(
                            device=gpu_id
                        )
                        self._stream_cache[
                            (sm, role)
                        ] = stream
                finally:
                    gc.pop_context()
                created_any = True
            except RuntimeError as e:
                if strict:
                    raise
                logger.warning(
                    "TorchNative: failed to create "
                    "green ctx with %d SMs: %s",
                    sm,
                    e,
                )
                continue

        if created_any:
            self._supported = True
            # Filter to only successfully created counts
            self._sm_counts = sorted(
                self._contexts.keys()
            )
            self._partition_sm = self._sm_counts[-1]
        else:
            self._reason = (
                "No green contexts could be created"
            )

    def is_supported(self) -> bool:
        return self._supported

    def unsupported_reason(self) -> str:
        return self._reason

    def available_sm_counts(self) -> List[int]:
        return list(self._sm_counts)

    def sm_step(self) -> int:
        return self._sm_step_val

    def partition_sm_count(self) -> int:
        return self._partition_sm

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        return StreamBundle(
            comp=self._stream_cache[
                (sm_count, "compute")
            ],
            recv=self._stream_cache[
                (sm_count, "recv")
            ],
            send=self._stream_cache[
                (sm_count, "send")
            ],
            dp=self._stream_cache[(sm_count, "dp")],
            sm_count=sm_count,
            generation=self._generation,
        )

    def activate_for_step(
        self, step_or_time: int
    ) -> Tuple[int, int]:
        # Determine target SM count from trace
        if self._trace_entries:
            sm = _sm_count_at_step(
                self._trace_entries,
                step_or_time,
                self._partition_sm,
            )
            # Snap to nearest available
            sm = min(
                self._sm_counts,
                key=lambda x: abs(x - sm),
            )
        else:
            sm = self._partition_sm

        # Pop current context if different
        if (
            self._active_gc is not None
            and self._active_sm != sm
        ):
            self._active_gc.pop_context()
            self._active_gc = None
            self._switch_count += 1
            self._generation += 1

        # Push new context if needed
        gc = self._contexts.get(sm)
        if gc is not None and self._active_gc is None:
            gc.set_context()
            self._active_gc = gc
            self._active_sm = sm
            if self._switch_count == 0:
                self._switch_count = 1
                self._generation = 1

        return sm, self._generation

    def activate_for_time(
        self, elapsed_us: int
    ) -> Tuple[int, int]:
        # Determine target SM count from trace
        if self._trace_entries:
            sm = _sm_count_at_time(
                self._trace_entries,
                elapsed_us,
                self._trace_time_unit,
                self._partition_sm,
            )
            # Snap to nearest available
            sm = min(
                self._sm_counts,
                key=lambda x: abs(x - sm),
            )
        else:
            sm = self._partition_sm

        # Pop current context if different
        if (
            self._active_gc is not None
            and self._active_sm != sm
        ):
            self._active_gc.pop_context()
            self._active_gc = None
            self._switch_count += 1
            self._generation += 1

        # Push new context if needed
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
        if self._active_gc is not None:
            self._active_gc.pop_context()
            self._active_gc = None

    def load_trace(self, path: str) -> bool:
        try:
            entries, tu, cm = _parse_trace_file(path)
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
        return _sm_count_at_time(
            self._trace_entries,
            elapsed_us,
            self._trace_time_unit,
            self._partition_sm,
        )

    def sm_count_at_step(self, step: int) -> int:
        if not self._trace_entries:
            return self._partition_sm
        return _sm_count_at_step(
            self._trace_entries,
            step,
            self._partition_sm,
        )

    def switch_count(self) -> int:
        return self._switch_count

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            # Pop active context if any
            if self._active_gc is not None:
                try:
                    self._active_gc.pop_context()
                except Exception:
                    pass
                self._active_gc = None
            self._stream_cache.clear()
            # GreenContext destructor handles cleanup
            self._contexts.clear()

# ── OffBackend ────────────────────────────────────────────


class OffBackend:
    """No-op backend using regular CUDA streams."""

    def __init__(
        self, gpu_id: int = 0, strict: bool = False
    ):
        self._gpu_id = gpu_id
        self._strict = strict
        self._closed = False

        if strict:
            raise RuntimeError(
                "Green contexts requested (strict=True) "
                "but no backend is available"
            )

        # Create regular streams (one set)
        with torch.cuda.device(gpu_id):
            self._streams = {
                "compute": torch.cuda.Stream(device=gpu_id),
                "recv": torch.cuda.Stream(device=gpu_id),
                "send": torch.cuda.Stream(device=gpu_id),
                "dp": torch.cuda.Stream(device=gpu_id),
            }
        self._bundle = StreamBundle(
            comp=self._streams["compute"],
            recv=self._streams["recv"],
            send=self._streams["send"],
            dp=self._streams["dp"],
            sm_count=0,
            generation=0,
        )

    def is_supported(self) -> bool:
        return False

    def unsupported_reason(self) -> str:
        return "Off backend (no green context)"

    def available_sm_counts(self) -> List[int]:
        return [0]

    def sm_step(self) -> int:
        return 0

    def partition_sm_count(self) -> int:
        return 0

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        return self._bundle

    def activate_for_step(
        self, step_or_time: int
    ) -> Tuple[int, int]:
        return 0, 0

    def activate_for_time(
        self, elapsed_us: int
    ) -> Tuple[int, int]:
        return 0, 0

    def deactivate(self, prev_sm_count: int) -> None:
        pass

    def load_trace(self, path: str) -> bool:
        warnings.warn(
            "OffBackend: trace ignored (no green context)",
            stacklevel=2,
        )
        return True

    def sm_count_at_time(self, elapsed_us: int) -> int:
        return 0

    def sm_count_at_step(self, step: int) -> int:
        return 0

    def switch_count(self) -> int:
        return 0

    def close(self) -> None:
        self._closed = True


# ── Backend selection ─────────────────────────────────────


def select_backend(
    backend: str = "auto",
    gpu_id: int = 0,
    strict: bool = False,
    **kwargs,
) -> GreenCtxBackend:
    """Select the best available backend.

    Priority: cpp > torch_native > off
    """
    if backend == "cpp" or backend == "auto":
        try:
            b = CppBackend(
                gpu_id=gpu_id, strict=strict, **kwargs
            )
            if b.is_supported():
                logger.info("Using CppBackend for green ctx")
                return b
            if backend == "cpp":
                if strict:
                    raise RuntimeError(
                        f"CppBackend not supported: "
                        f"{b.unsupported_reason()}"
                    )
                logger.warning(
                    "CppBackend requested but not supported: "
                    "%s. Falling back.",
                    b.unsupported_reason(),
                )
                b.close()
        except ImportError:
            if backend == "cpp":
                raise
            logger.debug(
                "morphling._GreenCtx not available, "
                "trying next backend"
            )

    if backend == "torch_native" or backend == "auto":
        b = TorchNativeBackend(
            gpu_id=gpu_id, strict=strict, **kwargs
        )
        if b.is_supported():
            logger.info(
                "Using TorchNativeBackend for green ctx"
            )
            return b
        if backend == "torch_native":
            if strict:
                raise RuntimeError(
                    f"TorchNativeBackend not supported: "
                    f"{b.unsupported_reason()}"
                )
            logger.warning(
                "TorchNativeBackend requested but not "
                "supported: %s",
                b.unsupported_reason(),
            )

    if backend == "off" or backend == "auto":
        logger.info("Using OffBackend (no green context)")
        return OffBackend(gpu_id=gpu_id, strict=strict)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        f"Expected: auto|cpp|torch_native|off"
    )
