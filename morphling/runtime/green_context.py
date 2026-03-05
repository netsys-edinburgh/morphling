"""GreenContextController — user-facing API for trace-driven
SM partitioning.

Usage:
    from morphling.runtime.green_context import (
        GreenContextController,
        GreenContextConfig,
    )

    cfg = GreenContextConfig(
        enabled=True,
        trace_path="traces/second_level.trace",
    )
    ctrl = GreenContextController.from_config(device_id=0, cfg=cfg)

    for step in range(num_steps):
        with ctrl.step_scope(step) as streams:
            with torch.cuda.stream(streams.comp):
                ...  # training step

    ctrl.close()
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional

import torch

from .green_context_backends import (
    GreenCtxBackend,
    OffBackend,
    StreamBundle,
    select_backend,
)

logger = logging.getLogger(__name__)


@dataclass
class GreenContextConfig:
    """Configuration for green context controller."""

    enabled: bool = False
    backend: str = "auto"  # auto|cpp|torch_native|off
    trace_path: Optional[str] = None
    clock_mode: str = "step"  # wall|step
    strict: bool = False
    switch_sync: str = "event_chain"
    num_partitions: int = 1
    partition_idx: int = 0
    stream_priority: int = -1


class GreenContextController:
    """Main user-facing controller for green context
    stream management.

    Provides:
    - step_scope(step_idx): context manager that activates
      the correct SM partition and yields pre-cached
      StreamBundle
    - Zero per-step overhead: no ExternalStream creation
      inside the hot path
    """

    def __init__(
        self,
        backend: GreenCtxBackend,
        config: GreenContextConfig,
    ):
        self._backend = backend
        self._config = config
        self._closed = False
        self._prev_sm: Optional[int] = None

        # Pre-cache stream bundles for all SM counts
        self._bundles: Dict[int, StreamBundle] = {}
        if backend.is_supported():
            for sm in backend.available_sm_counts():
                self._bundles[sm] = (
                    backend.get_stream_bundle(sm)
                )

        # Also cache for off backend
        if not backend.is_supported():
            self._bundles[0] = backend.get_stream_bundle(0)

    @classmethod
    def from_config(
        cls,
        device_id: int,
        cfg: GreenContextConfig,
    ) -> "GreenContextController":
        """Create controller from config."""
        if not cfg.enabled:
            backend = OffBackend(
                gpu_id=device_id, strict=False
            )
            return cls(backend=backend, config=cfg)

        backend = select_backend(
            backend=cfg.backend,
            gpu_id=device_id,
            strict=cfg.strict,
            num_partitions=cfg.num_partitions,
            partition_idx=cfg.partition_idx,
            stream_priority=cfg.stream_priority,
            switch_sync=cfg.switch_sync,
        )

        ctrl = cls(backend=backend, config=cfg)

        if cfg.trace_path and backend.is_supported():
            if not backend.load_trace(cfg.trace_path):
                if cfg.strict:
                    raise RuntimeError(
                        f"Failed to load trace: "
                        f"{cfg.trace_path}"
                    )
                logger.warning(
                    "Failed to load trace: %s",
                    cfg.trace_path,
                )

        return ctrl

    @property
    def is_supported(self) -> bool:
        return self._backend.is_supported()

    @property
    def backend(self) -> GreenCtxBackend:
        return self._backend

    @contextmanager
    def step_scope(
        self, step_idx: int
    ) -> Generator[StreamBundle, None, None]:
        """Context manager for one training step.

        Activates the SM partition for the given step,
        yields pre-cached streams, then restores on exit.

        Zero overhead: no ExternalStream creation inside
        this path.
        """
        sm, gen = self._backend.activate_for_step(step_idx)
        prev_sm = sm  # save for deactivate

        # Look up pre-cached bundle
        bundle = self._bundles.get(sm)
        if bundle is None or bundle.generation != gen:
            # Generation mismatch — rebuild just this bundle
            bundle = self._backend.get_stream_bundle(sm)
            self._bundles[sm] = bundle

        try:
            yield bundle
        finally:
            self._backend.deactivate(prev_sm)

    def get_default_bundle(self) -> StreamBundle:
        """Get the stream bundle for the default
        (full-partition) SM count."""
        if self._backend.is_supported():
            sm = self._backend.partition_sm_count()
        else:
            sm = 0
        return self._bundles[sm]

    def available_sm_counts(self):
        return self._backend.available_sm_counts()

    def activate_for_time(self, elapsed_us: int):
        return self._backend.activate_for_time(elapsed_us)

    def deactivate(self, prev_sm_count: int):
        return self._backend.deactivate(prev_sm_count)

    def switch_count(self) -> int:
        return self._backend.switch_count()

    def close(self) -> None:
        """Release all resources."""
        if not self._closed:
            self._closed = True
            self._bundles.clear()
            self._backend.close()

    def __del__(self):
        if not self._closed:
            self.close()
