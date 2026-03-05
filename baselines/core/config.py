from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast


@dataclass
class DeviceConfig:
    """Per-device runtime and hardware capability settings."""

    use_cuda: bool = True
    cuda_id: int = 0
    cuda_num: int = 1
    debug_mem: bool = True
    device_id: int = 0
    device_type: str = "gpu"
    memory_budget_mb: float = 4096.0
    compute_capacity: float = 1.0
    mps_enabled: bool = False
    mps_active_thread_percentage: int = 100
    mps_pipe_directory: str = ""
    mps_log_directory: str = ""
    mps_pinned_device_mem_limit: str = ""

    def validate(self) -> None:
        """Validate MPS configuration fields."""
        if not (1 <= self.mps_active_thread_percentage <= 100):
            msg = (
                "mps_active_thread_percentage must be 1-100, got "
                + str(self.mps_active_thread_percentage)
            )
            raise ValueError(
                msg
            )


@dataclass
class DistributedConfig:
    """Distributed-process and communication-group configuration."""

    dist_backend: str = "nccl"
    dist_url: str = "tcp://127.0.0.1:29500"
    world_size: int = 1
    pipeline_group_size: int = 1
    data_group_size: int = 1
    rank: int = 0
    timeout_s: float = 120.0


@dataclass
class ModelConfig:
    """Model architecture and task-head configuration."""

    model_name: str = "gpt2"
    model_type: str = "gpt2"
    task_type: str = "classification"
    task: str = "SeqClassification"
    seq_length: int = 2048
    max_seq_len: int = 2048
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 50257
    num_classes: int = 2
    dropout: float = 0.1
    use_flash_attention: bool = True

    def __post_init__(self) -> None:
        # YAML uses ``max_seq_len``; code paths use
        # ``seq_length``.  Keep them in sync.
        if self.max_seq_len != 2048 and self.seq_length == 2048:
            self.seq_length = self.max_seq_len
        elif self.seq_length != 2048 and self.max_seq_len == 2048:
            self.max_seq_len = self.seq_length


@dataclass
class TrainingConfig:
    """Optimization, batch sizing, and training-loop hyperparameters."""

    batch_size: int = 16
    global_batch_size: int = 16
    micro_batch_size: int = 4
    num_microbatches: int = 4
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: int = 50
    max_iters: int = 500
    num_iters: int = 500
    num_epochs: int = 1
    gradient_accumulate_step: int = 1
    seed: int = 42
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 100
    log_interval: int = 10
    betas: tuple[float, float] = (0.9, 0.95)

    def __post_init__(self) -> None:
        if self.global_batch_size < self.micro_batch_size:
            self.global_batch_size = self.micro_batch_size
        if self.batch_size <= 0:
            self.batch_size = self.global_batch_size
        if self.micro_batch_size > 0:
            self.num_microbatches = max(
                1,
                self.global_batch_size // self.micro_batch_size,
            )


@dataclass
class ParallelConfig:
    """Parallelism and schedule configuration for PP/DP execution."""

    pp_mode: str = "gpipe"
    dp_mode: str = "allreduce"
    gradient_accumulate_step: int = 1
    world_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    num_stages: int = 1
    schedule_type: str = "gpipe"
    comm_backend: str = "nccl"  # nccl | gloo | torch_dist
    d2d_bandwidth_mbps: float = 1000.0  # default bandwidth estimate (Mbps)

    def __post_init__(self) -> None:
        # Sync pp_size ↔ num_stages: whichever the user
        # set to > 1 wins.  The YAML typically has
        # ``num_stages`` while the dataclass default for
        # ``pp_size`` is 1, so derive the missing one.
        if self.num_stages > 1 and self.pp_size <= 1:
            self.pp_size = self.num_stages
        elif self.pp_size > 1 and self.num_stages <= 1:
            self.num_stages = self.pp_size
        elif self.num_stages <= 0:
            self.num_stages = max(self.pp_size, 1)


@dataclass
class FaultToleranceConfig:
    """Checkpoint, heartbeat, and recovery-policy configuration."""

    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 100
    heartbeat_interval_s: float = 5.0
    heartbeat_timeout_s: float = 15.0
    backward_timeout_ms: float = 30000.0
    replication_mode: str = "none"
    replication_interval: int = 50
    ft_check_interval: int = 10


@dataclass
class ClusterConfig:
    """Cluster topology configuration.

    NOTE: Bandwidth data is NOT stored here. All bandwidth measurements
    MUST come from iperf3 profiling during deployment. NO FALLBACKS.
    """

    # Reserved for future cluster-level config (not bandwidth)
    pass


@dataclass
class ParallelismPlan:
    """Planner output describing partitioning and micro-batch allocation."""

    partition_points: list[int] = field(default_factory=list)
    device_groups: dict[int, list[int]] = field(default_factory=dict)
    micro_batch_alloc: dict[int, dict[int, int]] = field(default_factory=dict)
    schedule_type: str = "gpipe"
    estimated_latency_ms: float = float("inf")
    node_mapping: dict[int, NodeInfo] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict for hpp_plan.json."""
        return {
            "partition_points": self.partition_points,
            "device_groups": {
                str(k): v
                for k, v in self.device_groups.items()
            },
            "micro_batch_alloc": {
                str(s): {
                    str(d): n
                    for d, n in alloc.items()
                }
                for s, alloc in self.micro_batch_alloc.items()
            },
            "schedule_type": self.schedule_type,
            "estimated_latency_ms": self.estimated_latency_ms,
            "node_mapping": {
                str(k): v.to_dict()
                for k, v in self.node_mapping.items()
            },
            "num_stages": len(self.partition_points) + 1,
            "world_size": sum(
                len(devs)
                for devs in self.device_groups.values()
            ),
        }

    @classmethod
    def from_json(
        cls, data: dict[str, object],
    ) -> ParallelismPlan:
        """Deserialize from hpp_plan.json dict."""
        node_mapping: dict[int, NodeInfo] = {}

        raw_node_mapping = data.get("node_mapping", {})
        if isinstance(raw_node_mapping, dict):
            for k, v in raw_node_mapping.items():
                if isinstance(v, dict):
                    node_mapping[int(k)] = NodeInfo.from_dict(
                        {
                            str(field_name): field_value
                            for field_name, field_value in v.items()
                        }
                    )

        partition_points_raw = data.get("partition_points", [])
        partition_points = (
            cast(list[int], partition_points_raw)
            if isinstance(partition_points_raw, list)
            else []
        )

        device_groups: dict[int, list[int]] = {}
        raw_device_groups = data.get("device_groups", {})
        if isinstance(raw_device_groups, dict):
            for k, v in raw_device_groups.items():
                if isinstance(v, list):
                    device_groups[int(k)] = [int(device) for device in v]

        micro_batch_alloc: dict[int, dict[int, int]] = {}
        raw_micro_batch_alloc = data.get("micro_batch_alloc", {})
        if isinstance(raw_micro_batch_alloc, dict):
            for stage, alloc in raw_micro_batch_alloc.items():
                if not isinstance(alloc, dict):
                    continue
                micro_batch_alloc[int(stage)] = {
                    int(device): int(batch_count)
                    for device, batch_count in alloc.items()
                }

        schedule_type = str(data.get("schedule_type", "gpipe"))
        estimated_raw = data.get("estimated_latency_ms", float("inf"))
        if isinstance(estimated_raw, (int, float)):
            estimated_latency_ms = float(estimated_raw)
        else:
            estimated_latency_ms = float("inf")

        return cls(
            partition_points=partition_points,
            device_groups=device_groups,
            micro_batch_alloc=micro_batch_alloc,
            schedule_type=schedule_type,
            estimated_latency_ms=estimated_latency_ms,
            node_mapping=node_mapping,
        )


@dataclass
class DeviceTopology:
    """Device-level topology and communication characteristics."""

    device_specs: list[DeviceConfig] = field(default_factory=list)
    bandwidths: dict[tuple[int, int], float] = field(default_factory=dict)
    latencies: dict[tuple[int, int], float] = field(default_factory=dict)


@dataclass
class NodeInfo:
    """Physical node information for K8s deployment."""

    hostname: str = "localhost"
    ip: str = "127.0.0.1"
    nic: str = "eth0"
    gpu_id: int = 0
    memory_mb: int = 4096
    architecture: str = "x86_64"

    def to_dict(self) -> dict[str, object]:
        return {
            "hostname": self.hostname,
            "ip": self.ip,
            "nic": self.nic,
            "gpu_id": self.gpu_id,
            "memory_mb": self.memory_mb,
            "architecture": self.architecture,
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, object],
    ) -> NodeInfo:
        gpu_id_raw = data.get("gpu_id", 0)
        memory_mb_raw = data.get("memory_mb", 4096)

        if isinstance(gpu_id_raw, (int, float, str)):
            gpu_id = int(gpu_id_raw)
        else:
            gpu_id = 0

        if isinstance(memory_mb_raw, (int, float, str)):
            memory_mb = int(memory_mb_raw)
        else:
            memory_mb = 4096

        return cls(
            hostname=str(data.get("hostname", "localhost")),
            ip=str(data.get("ip", "127.0.0.1")),
            nic=str(data.get("nic", "eth0")),
            gpu_id=gpu_id,
            memory_mb=memory_mb,
            architecture=str(data.get("architecture", "x86_64")),
        )


@dataclass
class GreenCtxConfig:
    """Green context SM partitioning configuration."""

    enabled: bool = False
    backend: str = "auto"  # auto|cpp|torch_native|off
    trace_path: str | None = None
    clock_mode: str = "step"  # wall|step
    strict: bool = False
    switch_sync: str = "event_chain"
    num_partitions: int = 1
    partition_idx: int = 0
    stream_priority: int = -1



@dataclass
class BaseConfig:
    """Unified baseline config combining model/training/runtime settings."""

    device: DeviceConfig = field(default_factory=DeviceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    fault_tolerance: FaultToleranceConfig = field(
        default_factory=FaultToleranceConfig,
    )
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    topology: DeviceTopology | None = None
    plan: ParallelismPlan | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    greenctx: GreenCtxConfig = field(
        default_factory=GreenCtxConfig,
    )


__all__ = [
    "BaseConfig",
    "ClusterConfig",
    "DeviceConfig",
    "DistributedConfig",
    "ModelConfig",
    "TrainingConfig",
    "ParallelConfig",
    "FaultToleranceConfig",
    "ParallelismPlan",
    "DeviceTopology",
    "NodeInfo",
    "GreenCtxConfig",
]
