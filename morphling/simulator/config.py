from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class ModelLayerConfig:
    name: str
    M: int
    K: int
    N: int
    param_bytes: int = 0

    @property
    def fwd_flops(self) -> float:
        return 2.0 * self.M * self.K * self.N

    @property
    def bwd_flops(self) -> float:
        # grad_input: 2*M*N*K + grad_weight: 2*M*K*N = 4*M*K*N
        return 4.0 * self.M * self.K * self.N

    @property
    def grad_bytes(self) -> int:
        return self.param_bytes if self.param_bytes else self.K * self.N * 4


@dataclass
class TransformerModelConfig:
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    intermediate_dim: int = 0
    vocab_size: int = 50257
    batch_size: int = 8
    seq_length: int = 512
    dtype_bytes: int = 4

    def to_layers(self) -> List[ModelLayerConfig]:
        h = self.hidden_dim
        inter = self.intermediate_dim or 4 * h
        M = self.batch_size * self.seq_length
        layers: List[ModelLayerConfig] = []

        for i in range(self.num_layers):
            prefix = f"layer_{i}"
            layers.append(
                ModelLayerConfig(
                    name=f"{prefix}.qkv",
                    M=M,
                    K=h,
                    N=3 * h,
                    param_bytes=h * 3 * h * self.dtype_bytes,
                )
            )
            layers.append(
                ModelLayerConfig(
                    name=f"{prefix}.attn_out",
                    M=M,
                    K=h,
                    N=h,
                    param_bytes=h * h * self.dtype_bytes,
                )
            )
            layers.append(
                ModelLayerConfig(
                    name=f"{prefix}.mlp_up",
                    M=M,
                    K=h,
                    N=inter,
                    param_bytes=h * inter * self.dtype_bytes,
                )
            )
            layers.append(
                ModelLayerConfig(
                    name=f"{prefix}.mlp_down",
                    M=M,
                    K=inter,
                    N=h,
                    param_bytes=inter * h * self.dtype_bytes,
                )
            )

        return layers

    @property
    def total_param_bytes(self) -> int:
        return sum(l.param_bytes for l in self.to_layers())


@dataclass
class DeviceSpec:
    rank: int
    flops: float
    ul_bw_bytes_per_s: float
    dl_bw_bytes_per_s: float
    ul_lat_s: float = 0.0
    dl_lat_s: float = 0.0

    @classmethod
    def from_morphling_config(cls, cfg) -> DeviceSpec:
        from morphling.common.config import human2bytes

        return cls(
            rank=cfg.rank,
            flops=float(human2bytes(str(cfg.flops))),
            ul_bw_bytes_per_s=float(human2bytes(str(cfg.ul_bw))),
            dl_bw_bytes_per_s=float(human2bytes(str(cfg.dl_bw))),
            ul_lat_s=float(cfg.ul_lat),
            dl_lat_s=float(cfg.dl_lat),
        )


@dataclass
class TopologyConfig:
    mode: Literal["allreduce", "ps", "hybrid"] = "allreduce"

    allreduce_algo: Literal["ring", "tree"] = "ring"
    ring_order: Optional[List[int]] = None

    num_ps_servers: int = 1
    ps_server_flops: float = 1e12
    ps_server_dl_bw: float = 10e9
    ps_server_ul_bw: float = 10e9

    intra_group_algo: Literal["allreduce", "ps"] = "allreduce"
    inter_group_algo: Literal["allreduce", "ps"] = "ps"
    device_groups: Optional[List[List[int]]] = None


@dataclass
class SimulationConfig:
    devices: List[DeviceSpec] = field(default_factory=list)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    layers: List[ModelLayerConfig] = field(default_factory=list)
    num_steps: int = 10
    overlap_mode: Literal["none", "full"] = "none"

    @classmethod
    def from_transformer(
        cls,
        model: TransformerModelConfig,
        devices: List[DeviceSpec],
        topology: TopologyConfig,
        num_steps: int = 10,
        overlap_mode: Literal["none", "full"] = "none",
    ) -> SimulationConfig:
        return cls(
            devices=devices,
            topology=topology,
            layers=model.to_layers(),
            num_steps=num_steps,
            overlap_mode=overlap_mode,
        )

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def total_param_bytes(self) -> int:
        return sum(l.param_bytes for l in self.layers)
