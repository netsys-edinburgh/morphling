"""Event data structures for Morphling simulator.

Defines frozen dataclasses for simulation events including capacity events,
reroute events, and reshard triggers with immutable slots for performance.
"""

# pyright: reportUntypedClassDecorator=false, reportUninitializedInstanceVariable=false
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional


@dataclass(frozen=True)
class CapacityEvent:
    __slots__: ClassVar[tuple[str, ...]] = (
        "timestamp_ns",
        "site_id",
        "spare_sms",
        "prev_spare_sms",
        "is_drop",
    )
    timestamp_ns: int
    site_id: int
    spare_sms: int
    prev_spare_sms: int
    is_drop: bool


@dataclass(frozen=True)
class RerouteEvent:
    __slots__: ClassVar[tuple[str, ...]] = (
        "timestamp_ns",
        "step_idx",
        "source_site",
        "target_site",
        "microbatches_moved",
        "stall_ns",
        "network_cost_ns",
        "decision",
    )
    timestamp_ns: int
    step_idx: int
    source_site: int
    target_site: int
    microbatches_moved: int
    stall_ns: int
    network_cost_ns: int
    decision: Literal["absorb", "defer"]


@dataclass(frozen=True)
class ReshardTrigger:
    __slots__: ClassVar[tuple[str, ...]] = (
        "timestamp_ns",
        "step_idx",
        "z_t",
        "theta_reshard",
        "d_t",
        "cooldown_remaining_steps",
    )
    timestamp_ns: int
    step_idx: int
    z_t: float
    theta_reshard: float
    d_t: float
    cooldown_remaining_steps: int


@dataclass(frozen=True)
class ReshardPhase:
    __slots__: ClassVar[tuple[str, ...]] = (
        "phase",
        "start_ns",
        "end_ns",
        "total_ns",
        "exposed_stall_ns",
        "hidden_ns",
    )
    phase: Literal["PREPARE", "COPY", "COMMIT"]
    start_ns: int
    end_ns: int
    total_ns: int
    exposed_stall_ns: int
    hidden_ns: int


@dataclass(frozen=True)
class ReshardEvent:
    __slots__: ClassVar[tuple[str, ...]] = (
        "trigger",
        "phases",
        "total_ns",
        "total_exposed_ns",
    )
    trigger: ReshardTrigger
    phases: list[ReshardPhase]
    total_ns: int
    total_exposed_ns: int


@dataclass(frozen=True)
class SchedulerStepState:
    __slots__: ClassVar[tuple[str, ...]] = (
        "step_idx",
        "timestamp_ns",
        "d_t",
        "z_t",
        "per_site_spare",
        "aggregate_spare",
        "level1_events",
        "level2_event",
        "step_latency_ns",
        "compute_ns",
        "comm_ns",
    )
    step_idx: int
    timestamp_ns: int
    d_t: float
    z_t: float
    per_site_spare: dict[int, int]
    aggregate_spare: int
    level1_events: list[RerouteEvent]
    level2_event: Optional[ReshardEvent]
    step_latency_ns: int
    compute_ns: int
    comm_ns: int


@dataclass(frozen=True)
class NetworkLink:
    __slots__: ClassVar[tuple[str, ...]] = (
        "source_site",
        "target_site",
        "bandwidth_gbps",
        "latency_ns",
    )
    source_site: int
    target_site: int
    bandwidth_gbps: float
    latency_ns: int


@dataclass(frozen=True)
class NetworkConfig:
    __slots__: ClassVar[tuple[str, ...]] = ("links",)
    links: list[NetworkLink]


@dataclass(frozen=True)
class NodeProfile:
    __slots__: ClassVar[tuple[str, ...]] = (
        "site_id",
        "node_type",
        "compute_sms",
        "step_compute_ns",
        "activation_bytes",
    )
    site_id: int
    node_type: str
    compute_sms: int
    step_compute_ns: int
    activation_bytes: int


@dataclass(frozen=True)
class ProfilingData:
    __slots__: ClassVar[tuple[str, ...]] = (
        "model_name",
        "nodes",
        "shard_bytes",
    )
    model_name: str
    nodes: list[NodeProfile]
    shard_bytes: int


@dataclass(frozen=True, init=False)
class SimulatorConfig:
    __slots__: ClassVar[tuple[str, ...]] = (
        "num_sites",
        "total_sms_per_site",
        "reshard_prepare_ns",
        "reshard_copy_ns",
        "reshard_commit_ns",
        "step_duration_ns",
        "ewma_alpha",
        "theta_reshard",
        "cooldown_steps",
        "reshard_overlap_ratio",
        "seed",
        "reroute_base_stall_ns",
        "reroute_per_mb_stall_ns",
    )
    num_sites: int
    total_sms_per_site: int
    reshard_prepare_ns: int
    reshard_copy_ns: int
    reshard_commit_ns: int
    step_duration_ns: int
    ewma_alpha: float
    theta_reshard: float
    cooldown_steps: int
    reshard_overlap_ratio: float
    seed: int
    reroute_base_stall_ns: int
    reroute_per_mb_stall_ns: int

    def __init__(
        self,
        num_sites: int,
        total_sms_per_site: int,
        reshard_prepare_ns: int,
        reshard_copy_ns: int,
        reshard_commit_ns: int,
        step_duration_ns: int,
        ewma_alpha: float = 0.3,
        theta_reshard: float = 0.7,
        cooldown_steps: int = 5,
        reshard_overlap_ratio: float = 0.75,
        seed: int = 42,
        reroute_base_stall_ns: int = 500_000,
        reroute_per_mb_stall_ns: int = 100_000,
    ) -> None:
        object.__setattr__(self, "num_sites", num_sites)
        object.__setattr__(self, "total_sms_per_site", total_sms_per_site)
        object.__setattr__(self, "reshard_prepare_ns", reshard_prepare_ns)
        object.__setattr__(self, "reshard_copy_ns", reshard_copy_ns)
        object.__setattr__(self, "reshard_commit_ns", reshard_commit_ns)
        object.__setattr__(self, "step_duration_ns", step_duration_ns)
        object.__setattr__(self, "ewma_alpha", ewma_alpha)
        object.__setattr__(self, "theta_reshard", theta_reshard)
        object.__setattr__(self, "cooldown_steps", cooldown_steps)
        object.__setattr__(self, "reshard_overlap_ratio", reshard_overlap_ratio)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "reroute_base_stall_ns", reroute_base_stall_ns)
        object.__setattr__(
            self, "reroute_per_mb_stall_ns", reroute_per_mb_stall_ns
        )
