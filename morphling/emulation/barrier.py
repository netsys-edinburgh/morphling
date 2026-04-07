"""Structural barrier with virtual time collection (threading backend)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BarrierResult:
    """Result of a structural barrier synchronization."""

    phase: str
    step_idx: int
    virtual_times_s: Dict[int, float]
    max_virtual_time_s: float
    wall_barrier_time_s: float


class StructuralBarrier:
    """Multi-device structural barrier with virtual time collection.

    Uses a triple-barrier pattern to guarantee:
    1. All threads deposit their virtual time.
    2. The elected thread builds the shared result visible to all.
    3. Cleanup completes before any thread re-enters.
    """

    def __init__(self, num_devices: int) -> None:
        self._num_devices = num_devices
        self._barrier = threading.Barrier(num_devices)
        self._lock = threading.Lock()
        self._virtual_times: Dict[int, float] = {}
        self._phase: str = ""
        self._step_idx: int = 0
        self._result: Optional[BarrierResult] = None

    def barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> BarrierResult:
        """Block until all devices arrive, then return collected virtual times."""
        wall_start = time.perf_counter()

        with self._lock:
            self._virtual_times[rank] = virtual_time_s
            self._phase = phase
            self._step_idx = step_idx

        elected = self._barrier.wait()

        if elected == 0:
            vt = dict(self._virtual_times)
            self._result = BarrierResult(
                phase=phase,
                step_idx=step_idx,
                virtual_times_s=vt,
                max_virtual_time_s=max(vt.values()),
                wall_barrier_time_s=0.0,
            )

        self._barrier.wait()

        result = self._result
        assert result is not None

        wall_elapsed = time.perf_counter() - wall_start
        per_thread_result = BarrierResult(
            phase=result.phase,
            step_idx=result.step_idx,
            virtual_times_s=result.virtual_times_s,
            max_virtual_time_s=result.max_virtual_time_s,
            wall_barrier_time_s=wall_elapsed,
        )

        if elected == 0:
            self._barrier.wait()
            with self._lock:
                self._virtual_times.clear()
            self._result = None
        else:
            self._barrier.wait()

        return per_thread_result
