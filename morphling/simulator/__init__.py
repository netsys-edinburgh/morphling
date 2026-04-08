"""Distributed training topology simulator for PS vs P2P comparison.

Simulates training iteration time under different communication topologies
(AllReduce, Parameter Server, Hybrid) on heterogeneous device configurations
using virtual timing. No actual GPU computation or data movement — pure
analytical cost modeling with causal event dependency tracking.
"""

from .config import (
    ModelLayerConfig,
    SimulationConfig,
    TopologyConfig,
    TransformerModelConfig,
)
from .runner import ComparisonRunner

__all__ = [
    "ModelLayerConfig",
    "SimulationConfig",
    "TopologyConfig",
    "TransformerModelConfig",
    "ComparisonRunner",
]
