"""BSP step executor with per-device time dilation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional

from morphling.simulator.config import TopologyConfig

from .barrier import StructuralBarrier
from .clock import DilatedDeviceClock
from .dilation import DilationMatrix


@dataclass
class EmulationStepResult:
    """Result of one BSP step in hybrid emulation mode."""

    step_idx: int
    wall_compute_s: Dict[int, float] = field(default_factory=dict)
    wall_comm_s: Dict[int, float] = field(default_factory=dict)
    virtual_compute_s: Dict[int, float] = field(default_factory=dict)
    virtual_comm_s: Dict[int, float] = field(default_factory=dict)
    virtual_compute_fwd_s: Dict[int, float] = field(default_factory=dict)
    virtual_compute_bwd_s: Dict[int, float] = field(default_factory=dict)
    virtual_total_s: Dict[int, float] = field(default_factory=dict)
    step_virtual_time_s: float = 0.0
    barrier_overhead_s: Dict[str, float] = field(default_factory=dict)

    def to_step_result(self):
        """Convert to PR #33 compatible StepResult."""
        from morphling.simulator.training_sim import StepResult

        ranks = sorted(self.virtual_total_s.keys())
        has_split = bool(self.virtual_compute_fwd_s)
        return StepResult(
            step_idx=self.step_idx,
            fwd_time_s={
                r: self.virtual_compute_fwd_s.get(r, 0.0)
                if has_split
                else self.virtual_compute_s.get(r, 0.0)
                for r in ranks
            },
            bwd_time_s={
                r: self.virtual_compute_bwd_s.get(r, 0.0) for r in ranks
            },
            grad_sync_time_s={
                r: self.virtual_comm_s.get(r, 0.0) for r in ranks
            },
            total_time_s=dict(self.virtual_total_s),
            step_time_s=self.step_virtual_time_s,
        )


@dataclass
class EmulationConfig:
    """Configuration for a hybrid emulation run."""

    dilation: DilationMatrix
    num_steps: int
    num_devices: int
    overlap_mode: Literal["none", "full"] = "none"
    topology: Optional[TopologyConfig] = None


class BSPStepRunner:
    """Per-device BSP step executor with dilation.

    One instance per device thread. Calls user-provided compute/comm
    callables, measures wall time, applies dilation, and synchronizes
    via structural barriers.
    """

    def __init__(
        self,
        rank: int,
        config: EmulationConfig,
        barrier: StructuralBarrier,
    ) -> None:
        self._rank = rank
        self._config = config
        self._barrier = barrier
        self._clock = DilatedDeviceClock(rank, config.dilation)

    def run_step(
        self,
        step_idx: int,
        compute_fn: Callable[[], None],
        comm_fn: Callable[[], None],
    ) -> EmulationStepResult:
        """Execute one BSP step: compute → barrier → comm → barrier → result."""
        self._clock.begin_step(step_idx)

        gemm_log_before = self._gemm_log_len()

        wall_start = time.perf_counter()
        compute_fn()
        wall_compute = time.perf_counter() - wall_start
        compute_rec = self._clock.record_compute(wall_compute)

        fwd_rec, bwd_rec = self._split_fwd_bwd(gemm_log_before, wall_compute)

        compute_br = self._barrier.barrier(
            self._rank, "compute", step_idx, self._clock.step_virtual_time_s
        )

        wall_start = time.perf_counter()
        comm_fn()
        wall_comm = time.perf_counter() - wall_start
        comm_rec = self._clock.record_comm(wall_comm)

        comm_br = self._barrier.barrier(
            self._rank, "comm", step_idx, self._clock.step_virtual_time_s
        )

        vt_compute = compute_rec.virtual_time_s
        vt_comm = comm_rec.virtual_time_s

        if self._config.overlap_mode == "full":
            vt_total = vt_compute + max(0.0, vt_comm - vt_compute)
        else:
            vt_total = vt_compute + vt_comm

        result = EmulationStepResult(step_idx=step_idx)
        result.wall_compute_s[self._rank] = wall_compute
        result.wall_comm_s[self._rank] = wall_comm
        result.virtual_compute_s[self._rank] = vt_compute
        result.virtual_comm_s[self._rank] = vt_comm
        if fwd_rec is not None:
            result.virtual_compute_fwd_s[self._rank] = fwd_rec.virtual_time_s
            result.virtual_compute_bwd_s[self._rank] = (
                bwd_rec.virtual_time_s if bwd_rec else 0.0
            )
        result.virtual_total_s[self._rank] = vt_total
        result.step_virtual_time_s = comm_br.max_virtual_time_s
        result.barrier_overhead_s["compute"] = compute_br.wall_barrier_time_s
        result.barrier_overhead_s["comm"] = comm_br.wall_barrier_time_s

        return result

    @staticmethod
    def _gemm_log_len() -> int:
        try:
            from morphling.hooks.autograd import get_gemm_log

            return len(get_gemm_log())
        except (ImportError, AttributeError):
            return -1

    def _split_fwd_bwd(self, log_before: int, wall_compute: float) -> tuple:
        if log_before < 0:
            return None, None
        try:
            from morphling.hooks.autograd import get_gemm_log

            entries = get_gemm_log()[log_before:]
        except (ImportError, AttributeError):
            return None, None

        if not entries:
            return None, None

        fwd_us = sum(
            e["duration_us"] for e in entries if e["phase"] == "forward"
        )
        bwd_us = sum(
            e["duration_us"] for e in entries if "backward" in e["phase"]
        )
        total_us = fwd_us + bwd_us
        if total_us <= 0:
            return None, None

        wall_fwd = wall_compute * (fwd_us / total_us)
        wall_bwd = wall_compute * (bwd_us / total_us)

        fwd_rec = self._clock.record_compute_fwd(wall_fwd)
        bwd_rec = self._clock.record_compute_bwd(wall_bwd)
        return fwd_rec, bwd_rec
