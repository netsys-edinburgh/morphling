"""Per-device virtual clock with phase-aware dilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .dilation import DilationMatrix


@dataclass
class PhaseTimingRecord:
    """Wall and virtual timing for one phase of one step on one device."""

    phase: str
    wall_time_s: float
    virtual_time_s: float
    dilation_factor: float


class DilatedDeviceClock:
    """Tracks virtual time for a single device across BSP step phases."""

    def __init__(self, rank: int, dilation: DilationMatrix) -> None:
        self._rank = rank
        self._dilation = dilation
        self._step_idx: int | None = None
        self._step_phases: List[PhaseTimingRecord] = []
        self._cumulative_virtual_s: float = 0.0

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def step_virtual_time_s(self) -> float:
        """Virtual time accumulated in the current step only."""
        return sum(p.virtual_time_s for p in self._step_phases)

    def total_virtual_time_s(self) -> float:
        """Cumulative virtual time across ALL completed steps plus current."""
        return self._cumulative_virtual_s + self.step_virtual_time_s

    def begin_step(self, step_idx: int) -> None:
        """Finalize previous step and start a new one."""
        self._cumulative_virtual_s += self.step_virtual_time_s
        self._step_phases = []
        self._step_idx = step_idx

    def record_compute(self, wall_time_s: float) -> PhaseTimingRecord:
        """Record a compute phase with alpha dilation."""
        self._check_step_started()
        alpha = self._dilation.alpha(self._rank)
        rec = PhaseTimingRecord(
            phase="compute",
            wall_time_s=wall_time_s,
            virtual_time_s=wall_time_s * alpha,
            dilation_factor=alpha,
        )
        self._step_phases.append(rec)
        return rec

    def record_compute_fwd(self, wall_time_s: float) -> PhaseTimingRecord:
        """Record forward compute phase with alpha dilation."""
        self._check_step_started()
        alpha = self._dilation.alpha(self._rank)
        rec = PhaseTimingRecord(
            phase="compute_fwd",
            wall_time_s=wall_time_s,
            virtual_time_s=wall_time_s * alpha,
            dilation_factor=alpha,
        )
        self._step_phases.append(rec)
        return rec

    def record_compute_bwd(self, wall_time_s: float) -> PhaseTimingRecord:
        """Record backward compute phase with alpha dilation."""
        self._check_step_started()
        alpha = self._dilation.alpha(self._rank)
        rec = PhaseTimingRecord(
            phase="compute_bwd",
            wall_time_s=wall_time_s,
            virtual_time_s=wall_time_s * alpha,
            dilation_factor=alpha,
        )
        self._step_phases.append(rec)
        return rec

    def record_comm(self, wall_time_s: float) -> PhaseTimingRecord:
        """Record a communication phase with beta dilation."""
        self._check_step_started()
        beta = self._dilation.beta(self._rank)
        rec = PhaseTimingRecord(
            phase="comm",
            wall_time_s=wall_time_s,
            virtual_time_s=wall_time_s * beta,
            dilation_factor=beta,
        )
        self._step_phases.append(rec)
        return rec

    def step_summary(self) -> Dict[str, float]:
        """Return ``{phase: virtual_time_s}`` for the current step."""
        out: Dict[str, float] = {}
        for p in self._step_phases:
            out[p.phase] = out.get(p.phase, 0.0) + p.virtual_time_s
        return out

    def _check_step_started(self) -> None:
        if self._step_idx is None:
            raise RuntimeError(
                "begin_step() must be called before recording phases"
            )
