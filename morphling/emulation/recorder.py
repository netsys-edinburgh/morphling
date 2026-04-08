"""Collects hybrid emulation results into SimulationResult format."""

from __future__ import annotations

import statistics
from typing import Dict, List

from morphling.simulator.training_sim import SimulationResult, StepResult

from .step_runner import EmulationStepResult


class EmulationRecorder:
    """Collects per-step emulation results and produces SimulationResult."""

    def __init__(self, topology_name: str = "emulation") -> None:
        self._topology_name = topology_name
        self._steps: List[EmulationStepResult] = []
        self._finalized = False

    def record_step(self, step_result: EmulationStepResult) -> None:
        """Add a completed step's results."""
        if self._finalized:
            raise RuntimeError("Cannot record after finalize()")
        self._steps.append(step_result)

    def finalize(self) -> SimulationResult:
        """Compute aggregates and return SimulationResult."""
        if self._finalized:
            raise RuntimeError("finalize() already called")
        self._finalized = True

        pr33_steps: List[StepResult] = []
        step_times: List[float] = []

        for emu_step in self._steps:
            pr33_steps.append(emu_step.to_step_result())
            step_times.append(emu_step.step_virtual_time_s)

        mean_step = statistics.fmean(step_times) if step_times else 0.0
        std_step = statistics.pstdev(step_times) if len(step_times) > 1 else 0.0
        throughput = (1.0 / mean_step) if mean_step > 0.0 else 0.0

        return SimulationResult(
            topology=self._topology_name,
            steps=pr33_steps,
            mean_step_time_s=mean_step,
            std_step_time_s=std_step,
            throughput_steps_per_s=throughput,
        )

    def wall_time_summary(self) -> Dict[str, float]:
        """Return wall-time statistics for overhead analysis."""
        total_wall = 0.0
        total_barrier = 0.0
        for s in self._steps:
            for wc in s.wall_compute_s.values():
                total_wall += wc
            for wm in s.wall_comm_s.values():
                total_wall += wm
            for bo in s.barrier_overhead_s.values():
                total_barrier += bo

        return {
            "total_wall_time_s": total_wall,
            "total_barrier_overhead_s": total_barrier,
            "barrier_overhead_pct": (
                100.0 * total_barrier / total_wall if total_wall > 0 else 0.0
            ),
        }
