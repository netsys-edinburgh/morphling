from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .config import SimulationConfig, TopologyConfig
from .cost_models import compute_time_s, gradient_sync_time_s


@dataclass
class StepResult:
    step_idx: int
    fwd_time_s: dict[int, float] = field(default_factory=dict)
    bwd_time_s: dict[int, float] = field(default_factory=dict)
    grad_sync_time_s: dict[int, float] = field(default_factory=dict)
    total_time_s: dict[int, float] = field(default_factory=dict)
    step_time_s: float = 0.0


@dataclass
class SimulationResult:
    """Aggregated simulation output for one training topology."""

    topology: str
    steps: list[StepResult] = field(default_factory=list)
    mean_step_time_s: float = 0.0
    std_step_time_s: float = 0.0
    throughput_steps_per_s: float = 0.0

    def communication_overhead_pct(self) -> float:
        total_comm = 0.0
        total_time = 0.0
        for step in self.steps:
            total_comm += sum(step.grad_sync_time_s.values())
            total_time += sum(step.total_time_s.values())
        if total_time <= 0.0:
            return 0.0
        return 100.0 * total_comm / total_time

    def slowest_device_rank(self) -> int | None:
        totals_by_rank: dict[int, float] = {}
        counts_by_rank: dict[int, int] = {}
        for step in self.steps:
            for rank, total in step.total_time_s.items():
                totals_by_rank[rank] = totals_by_rank.get(rank, 0.0) + total
                counts_by_rank[rank] = counts_by_rank.get(rank, 0) + 1
        if not totals_by_rank:
            return None
        mean_by_rank = {
            rank: totals_by_rank[rank] / counts_by_rank[rank]
            for rank in totals_by_rank
        }
        return max(mean_by_rank, key=lambda rank: mean_by_rank[rank])


def topology_description(topo: TopologyConfig) -> str:
    if topo.mode == "allreduce":
        return f"allreduce-{topo.allreduce_algo}"
    if topo.mode == "ps":
        suffix = "server" if topo.num_ps_servers == 1 else "servers"
        return f"ps-{topo.num_ps_servers}{suffix}"
    intra = "ar" if topo.intra_group_algo == "allreduce" else "ps"
    inter = "ar" if topo.inter_group_algo == "allreduce" else "ps"
    return f"hybrid-{intra}-{inter}"


class TrainingSimulator:
    """Simulates BSP training steps for a fixed model, device set, and topology."""

    config: SimulationConfig

    def __init__(self, config: SimulationConfig):
        self.config = config

    def simulate(self, num_steps: int | None = None) -> SimulationResult:
        effective_steps = (
            self.config.num_steps if num_steps is None else num_steps
        )
        steps = [
            self._simulate_step(step_idx) for step_idx in range(effective_steps)
        ]
        step_times = [step.step_time_s for step in steps]

        mean_step = statistics.fmean(step_times) if step_times else 0.0
        std_step = statistics.pstdev(step_times) if len(step_times) > 1 else 0.0
        throughput = (1.0 / mean_step) if mean_step > 0.0 else 0.0

        return SimulationResult(
            topology=topology_description(self.config.topology),
            steps=steps,
            mean_step_time_s=mean_step,
            std_step_time_s=std_step,
            throughput_steps_per_s=throughput,
        )

    def _simulate_step(self, step_idx: int) -> StepResult:
        devices = self.config.devices
        fwd_by_rank = {device.rank: 0.0 for device in devices}
        bwd_by_rank = {device.rank: 0.0 for device in devices}
        grad_by_rank = {device.rank: 0.0 for device in devices}

        for layer in self.config.layers:
            for device in devices:
                fwd_by_rank[device.rank] += compute_time_s(
                    layer.fwd_flops, device.flops
                )
                bwd_by_rank[device.rank] += compute_time_s(
                    layer.bwd_flops, device.flops
                )

            grad_times = gradient_sync_time_s(
                layer.grad_bytes,
                devices,
                self.config.topology,
            )
            for rank, grad_time in grad_times.items():
                grad_by_rank[rank] += grad_time

        total_by_rank: dict[int, float] = {}
        for device in devices:
            rank = device.rank
            fwd = fwd_by_rank[rank]
            bwd = bwd_by_rank[rank]
            grad = grad_by_rank[rank]
            if self.config.overlap_mode == "full":
                total = fwd + max(bwd, grad)
            else:
                total = fwd + bwd + grad
            total_by_rank[rank] = total

        step_time = max(total_by_rank.values(), default=0.0)
        return StepResult(
            step_idx=step_idx,
            fwd_time_s=fwd_by_rank,
            bwd_time_s=bwd_by_rank,
            grad_sync_time_s=grad_by_rank,
            total_time_s=total_by_rank,
            step_time_s=step_time,
        )
