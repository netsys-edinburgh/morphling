from __future__ import annotations

from typing import Literal

from .config import (
    DeviceSpec,
    ModelLayerConfig,
    SimulationConfig,
    TopologyConfig,
)
from .training_sim import (
    SimulationResult,
    TrainingSimulator,
    topology_description,
)


class ComparisonRunner:
    """Runs identical training simulation across multiple topology settings."""

    layers: list[ModelLayerConfig]
    devices: list[DeviceSpec]
    topologies: list[TopologyConfig]
    num_steps: int
    overlap_mode: Literal["none", "full"]

    def __init__(
        self,
        layers: list[ModelLayerConfig],
        devices: list[DeviceSpec],
        topologies: list[TopologyConfig],
        num_steps: int = 10,
        overlap_mode: Literal["none", "full"] = "none",
    ):
        self.layers = layers
        self.devices = devices
        self.topologies = topologies
        self.num_steps = num_steps
        self.overlap_mode = overlap_mode

    def run(self) -> dict[str, SimulationResult]:
        results: dict[str, SimulationResult] = {}
        for topology in self.topologies:
            sim_cfg = SimulationConfig(
                devices=self.devices,
                topology=topology,
                layers=self.layers,
                num_steps=self.num_steps,
                overlap_mode=self.overlap_mode,
            )
            simulator = TrainingSimulator(sim_cfg)
            result = simulator.simulate(self.num_steps)
            results[topology_description(topology)] = result
        return results
