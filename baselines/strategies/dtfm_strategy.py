from __future__ import annotations

import logging
from typing_extensions import override

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    ParallelismPlan,
)
from baselines.core.profiler import ProfilerBackend
from baselines.core.scheduler import SchedulerBackend

try:
    from baselines.schedulers.gcma import (
        GCMAScheduler,
    )

    _HAS_GCMA = True
except ImportError:
    _HAS_GCMA = False

from .base import ParallelismStrategy

logger = logging.getLogger(__name__)


class DTFMStrategy(ParallelismStrategy):
    """DT-FM baseline with GCMA topology search and DP partitioning."""

    def __init__(
        self,
        pp_size: int,
        dp_size: int,
        population_size: int = 100,
        gcma_trails: int = 4900,
        global_batch_size: int = 64,
        micro_batch_size: int = 2,
    ) -> None:
        self.pp_size: int = max(1, pp_size)
        self.dp_size: int = max(1, dp_size)
        self.population_size: int = max(2, population_size)
        self.gcma_trails: int = max(1, gcma_trails)
        self.global_batch_size: int = max(1, global_batch_size)
        self.micro_batch_size: int = max(1, micro_batch_size)
        self.parallel_config: ParallelConfig = ParallelConfig(
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            schedule_type=self.get_schedule_type(),
        )
        self.scheduler_backend: SchedulerBackend | None = None

    @override
    def create_plan(
        self,
        model_config: ModelConfig,
        device_topology: DeviceTopology,
        profiler: ProfilerBackend | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        num_devices = len(topology.device_specs)
        num_layers = max(1, model_config.num_layers)

        best_plan: ParallelismPlan | None = None
        best_eff: float = float("inf")

        # Auto-search every valid (pp, dp) factorisation
        # of num_devices and pick lowest pipeline latency
        # (ported from asteroid_project).
        # DTFM by definition does pipeline partitioning
        # (GCMA + DP split); pp=1 would be pure DP with
        # no pipeline, defeating the purpose.  The paper
        # only considers pp >= 2.  We also avoid pp=1
        # because the schedule_factor does not account
        # for allreduce cost, making pp=1 always win
        # trivially on homogeneous hardware.
        min_pp = 2 if num_devices > 1 else 1
        for pp in range(
            min_pp, min(num_devices, num_layers) + 1
        ):
            if num_devices % pp != 0:
                continue
            dp = num_devices // pp

            saved_pp, saved_dp = (
                self.pp_size,
                self.dp_size,
            )
            self.pp_size, self.dp_size = pp, dp
            try:
                if num_devices > 1 and pp > 1:
                    device_groups = self._run_gcma(
                        topology, model_config
                    )
                else:
                    device_ids = [
                        s.device_id
                        for s in topology.device_specs
                    ]
                    device_groups = {0: device_ids}

                pts, lat = self._equal_partition(
                    model_config,
                    topology,
                    device_groups,
                    profiler,
                )
            finally:
                self.pp_size, self.dp_size = (
                    saved_pp,
                    saved_dp,
                )

            if lat == float("inf") or lat < 0:
                continue

            # Full schedule cost: bottleneck × (PP +
            # ceil(M/dp) - 1), accounting for pipeline
            # fill/drain overhead.
            M = max(
                1,
                self.global_batch_size
                // self.micro_batch_size,
            )
            micro_per_pipe = max(1, -(-M // dp))
            schedule_factor = pp + micro_per_pipe - 1
            eff = lat * schedule_factor

            if eff < best_eff:
                best_eff = eff
                best_plan = ParallelismPlan(
                    partition_points=pts,
                    device_groups=device_groups,
                    micro_batch_alloc={},
                    schedule_type=self.get_schedule_type(),
                    estimated_latency_ms=eff,
                )
                logger.debug(
                    "DTFM candidate pp=%d dp=%d "
                    "step=%.2fms batch=%.2fms",
                    pp,
                    dp,
                    lat,
                    eff,
                )

        if best_plan is None:
            device_ids = [
                s.device_id
                for s in topology.device_specs
            ]
            best_plan = ParallelismPlan(
                device_groups={0: device_ids},
                schedule_type=self.get_schedule_type(),
            )

        logger.info(
            "DTFM strategy plan points=%s latency=%.2fms",
            best_plan.partition_points,
            best_plan.estimated_latency_ms,
        )
        return best_plan

    @override
    def get_schedule_type(self) -> str:
        return "1f1b"

    @override
    def get_fault_tolerance_config(self) -> FaultToleranceConfig:
        return FaultToleranceConfig(
            checkpoint_dir="./checkpoints/dtfm",
            checkpoint_interval=100,
            heartbeat_interval_s=0.0,
            heartbeat_timeout_s=0.0,
            backward_timeout_ms=0.0,
            replication_mode="none",
            replication_interval=0,
            ft_check_interval=10,
        )

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        if topology.device_specs:
            return topology
        count = max(1, self.pp_size * self.dp_size)
        specs = [DeviceConfig(device_id=i) for i in range(count)]
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    def _run_gcma(
        self,
        topology: DeviceTopology,
        model_config: ModelConfig,
    ) -> dict[int, list[int]]:
        """Delegate to the paper-faithful GCMAScheduler.

        Builds numpy peer_delay / peer_bandwidth matrices from
        the DeviceTopology and converts the resulting gpu_map
        back into ``{stage_idx: [device_ids]}``.
        """
        specs = topology.device_specs
        device_ids = [
            spec.device_id for spec in specs
        ]
        num_devices = len(device_ids)
        stage_count = max(
            1, min(self.pp_size, num_devices)
        )
        if stage_count == 1:
            return {0: device_ids}

        dp_size = max(1, num_devices // stage_count)

        if not _HAS_NUMPY or not _HAS_GCMA:
            logger.warning(
                "numpy/GCMAScheduler unavailable; "
                "falling back to round-robin "
                "device assignment"
            )
            groups: dict[int, list[int]] = {
                s: [] for s in range(stage_count)
            }
            for idx, did in enumerate(device_ids):
                groups[idx % stage_count].append(
                    did
                )
            return groups

        # Build NxN peer matrices (indexed by
        # position in device_ids, not device_id)
        id_to_idx = {
            did: idx
            for idx, did in enumerate(device_ids)
        }
        peer_delay = np.zeros(
            (num_devices, num_devices),
            dtype=float,
        )
        peer_bandwidth = np.ones(
            (num_devices, num_devices),
            dtype=float,
        )
        for i, src in enumerate(specs):
            for j, dst in enumerate(specs):
                if i == j:
                    # Self-links: zero latency, effectively infinite BW
                    peer_delay[i, j] = 0.0
                    peer_bandwidth[i, j] = 100.0  # 100 Gbps local
                    continue
                delay = self._lookup_metric_strict(
                    topology.latencies,
                    src.device_id,
                    dst.device_id,
                    "latency",
                )
                bw_mbps = self._lookup_metric_strict(
                    topology.bandwidths,
                    src.device_id,
                    dst.device_id,
                    "bandwidth",
                )
                peer_delay[i, j] = max(delay, 1e-6)
                # GCMAScheduler expects Gbps
                peer_bandwidth[i, j] = max(
                    bw_mbps / 1000.0, 1e-6
                )

        # Compute gradient and activation sizes (GB)
        # from model architecture via ModelConfig helpers.
        #
        # gradient  = params_per_stage × 4 bytes / 1 GB
        # activation = micro_batch × seq_len × embed_dim × 4 / 1 GB
        send_gradient_gb = model_config.gradient_size_gb(
            num_stages=stage_count,
            bytes_per_param=4,
        )
        send_activation_gb = model_config.activation_size_gb(
            bytes_per_element=4,
        )

        gcma = GCMAScheduler(
            num_devices=num_devices,
            pp_size=stage_count,
            dp_size=dp_size,
            peer_delay=peer_delay,
            peer_bandwidth=peer_bandwidth,
            send_gradient_size=send_gradient_gb,
            send_activation_size=(
                send_activation_gb
            ),
        )
        result = gcma.solve()

        gpu_map = result.get("gpu_map", {})
        if not isinstance(gpu_map, dict):
            gpu_map = {}

        total_cost = result.get(
            "total_cost", float("inf")
        )
        logger.debug(
            "GCMA (paper) total cost %.6f",
            total_cost,
        )

        # Convert gpu_map (global_rank → matrix idx)
        # to device_groups (stage → [device_ids]).
        # DT-FM rank layout:
        #   global_rank = dp_rank * pp_size + pp_rank
        # So pp_rank = global_rank % pp_size  (stage)
        groups: dict[int, list[int]] = {
            s: [] for s in range(stage_count)
        }
        for global_rank, matrix_idx in (
            gpu_map.items()
        ):
            pp_rank = int(global_rank) % stage_count
            # matrix_idx is positional; map back
            # to actual device_id
            if 0 <= matrix_idx < num_devices:
                did = device_ids[matrix_idx]
            else:
                did = int(matrix_idx)
            groups[pp_rank].append(did)

        # Ensure no empty stages
        for s in range(stage_count):
            if not groups[s]:
                groups[s] = [device_ids[s % num_devices]]

        return groups

    def _lookup_metric_strict(
        self,
        table: dict[tuple[int, int], float],
        src: int,
        dst: int,
        metric_name: str,
    ) -> float:
        """Lookup metric - raises error if not found.

        NO FALLBACK VALUES ALLOWED.
        """
        direct = table.get((src, dst))
        if direct is not None:
            return float(direct)
        reverse = table.get((dst, src))
        if reverse is not None:
            return float(reverse)
        raise RuntimeError(
            f"Missing {metric_name} for link ({src}, {dst}). "
            f"Re-run profiling with iperf3 on ALL nodes. "
            f"NO FALLBACK VALUES ALLOWED."
        )

    def _equal_partition(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        device_groups: dict[int, list[int]],
        profiler: ProfilerBackend | None,
    ) -> tuple[list[int], float]:
        """Paper-faithful equal layer partition: L / D_PP.

        The DTFM paper specifies equal partitioning of
        layers across pipeline stages, not heterogeneous
        DP-based partitioning.
        """
        num_layers = max(1, model_config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = min(len(stage_ids), num_layers)

        if num_stages <= 1:
            # Single stage — same as _dp_partition
            spec_by_id = {
                spec.device_id: spec
                for spec in topology.device_specs
            }
            group = device_groups.get(
                stage_ids[0] if stage_ids else 0,
                [],
            )
            if not group:
                return [], 0.0
            worst_total = 0.0
            for did in group:
                sp = spec_by_id.get(
                    did,
                    DeviceConfig(device_id=did),
                )
                total = sum(
                    self._layer_time(
                        li, did, sp,
                        model_config, profiler,
                    )
                    for li in range(num_layers)
                )
                worst_total = max(
                    worst_total, total
                )
            return [], worst_total

        # Equal split: each stage gets L // D_PP layers,
        # remainder distributed to first stages.
        base = num_layers // num_stages
        remainder = num_layers % num_stages
        partition_points: list[int] = []
        cum = 0
        for s in range(num_stages - 1):
            cum += base + (1 if s < remainder else 0)
            partition_points.append(cum)

        # Compute bottleneck latency across stages
        spec_by_id = {
            spec.device_id: spec
            for spec in topology.device_specs
        }
        boundaries = [0] + partition_points + [num_layers]
        bottleneck = 0.0
        for s in range(num_stages):
            start_layer = boundaries[s]
            end_layer = boundaries[s + 1]
            stage = stage_ids[s]
            group = device_groups[stage]
            # Bottleneck device in group
            worst_t = 0.0
            for did in group:
                sp = spec_by_id.get(
                    did,
                    DeviceConfig(device_id=did),
                )
                t = sum(
                    self._layer_time(
                        li, did, sp,
                        model_config, profiler,
                    )
                    for li in range(
                        start_layer, end_layer
                    )
                )
                worst_t = max(worst_t, t)

            # Add inter-stage comm for non-last stages
            if s < num_stages - 1:
                comm = self._boundary_comm_time(
                    boundary_layer=end_layer - 1,
                    left_group=group,
                    right_group=device_groups[
                        stage_ids[s + 1]
                    ],
                    model_config=model_config,
                    topology=topology,
                    profiler=profiler,
                )
                worst_t += comm

            bottleneck = max(bottleneck, worst_t)

        return partition_points, bottleneck

    def _dp_partition(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        device_groups: dict[int, list[int]],
        profiler: ProfilerBackend | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, model_config.num_layers)
        stage_ids = sorted(device_groups)
        num_stages = min(len(stage_ids), num_layers)
        spec_by_id = {
            spec.device_id: spec
            for spec in topology.device_specs
        }
        if num_stages <= 1:
            # Single stage — compute actual total
            # latency on the bottleneck device so the
            # auto-search can compare it fairly against
            # multi-stage plans (returning 0.0 would
            # make pp=1 always win trivially).
            group = device_groups.get(
                stage_ids[0] if stage_ids else 0,
                [],
            )
            if not group:
                return [], 0.0
            worst_total = 0.0
            for did in group:
                sp = spec_by_id.get(
                    did,
                    DeviceConfig(device_id=did),
                )
                total = sum(
                    self._layer_time(
                        li, did, sp,
                        model_config, profiler,
                    )
                    for li in range(num_layers)
                )
                worst_total = max(
                    worst_total, total
                )
            return [], worst_total

        stage_prefix: list[list[float]] = []
        for stage_idx in range(num_stages):
            stage = stage_ids[stage_idx]
            group = device_groups[stage]
            # Use the slowest (bottleneck) device in the
            # group as representative — matches
            # asteroid_project.
            rep_id = group[0]
            rep_time = float("-inf")
            for did in group:
                sp = spec_by_id.get(
                    did, DeviceConfig(device_id=did)
                )
                t = sum(
                    self._layer_time(
                        li, did, sp, model_config,
                        profiler,
                    )
                    for li in range(num_layers)
                )
                if t > rep_time:
                    rep_time = t
                    rep_id = did
            spec = spec_by_id.get(rep_id, DeviceConfig(device_id=rep_id))
            prefix = [0.0]
            for layer_idx in range(num_layers):
                layer_t = self._layer_time(
                    layer_idx,
                    rep_id,
                    spec,
                    model_config,
                    profiler,
                )
                # _layer_time already incorporates capacity
                # via cap_factor; do NOT divide again.
                prefix.append(prefix[-1] + layer_t)
            stage_prefix.append(prefix)

        dp = [
            [float("inf")] * num_stages
            for _ in range(num_layers)
        ]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(stage_idx: int, start: int, end: int) -> float:
            prefix = stage_prefix[stage_idx]
            return prefix[end + 1] - prefix[start]

        for i in range(num_layers):
            dp[i][0] = range_cost(0, 0, i)

        for stage_idx in range(1, num_stages):
            for end in range(stage_idx, num_layers):
                for cut in range(stage_idx - 1, end):
                    comp = range_cost(stage_idx, cut + 1, end)
                    comm = self._boundary_comm_time(
                        boundary_layer=cut,
                        left_group=device_groups[stage_ids[stage_idx - 1]],
                        right_group=device_groups[stage_ids[stage_idx]],
                        model_config=model_config,
                        topology=topology,
                        profiler=profiler,
                    )
                    candidate = max(dp[cut][stage_idx - 1], comp + comm)
                    if candidate < dp[end][stage_idx]:
                        dp[end][stage_idx] = candidate
                        split[end][stage_idx] = cut

        points: list[int] = []
        end = num_layers - 1
        stage_idx = num_stages - 1
        while stage_idx > 0:
            cut = split[end][stage_idx]
            if cut < 0:
                break
            # cut is the inclusive end index of the previous stage.
            # train.py uses exclusive boundaries, so store cut + 1.
            points.append(cut + 1)
            end = cut
            stage_idx -= 1
        points.reverse()
        bottleneck = dp[num_layers - 1][num_stages - 1]
        if bottleneck == float("inf"):
            raise RuntimeError(
                f"DP partition failed: no valid partition found. "
                f"num_layers={num_layers} num_stages={num_stages}. "
                f"Ensure all layer timing and bandwidth data are profiled. "
                f"NO FALLBACK VALUES ALLOWED."
            )
        return points, bottleneck

    def _layer_time(
        self,
        layer_idx: int,
        device_id: int,
        device_cfg: DeviceConfig,
        model_config: ModelConfig,
        profiler: ProfilerBackend | None,
    ) -> float:
        if profiler is None:
            raise RuntimeError(
                f"Profiler required for layer timing. "
                f"device={device_id} layer={layer_idx}. "
                f"NO FALLBACK VALUES ALLOWED."
            )
        try:
            fwd = profiler.get_time_interval(
                device_id,
                layer_idx,
                layer_idx,
                0,
            )
            bwd = profiler.get_time_interval(
                device_id,
                layer_idx,
                layer_idx,
                1,
            )
            if fwd > 0.0 and bwd > 0.0:
                return fwd + bwd
            raise RuntimeError(
                f"Invalid profiler timing for device={device_id} "
                f"layer={layer_idx}: fwd={fwd} bwd={bwd}. "
                f"Re-run profiling phase."
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Profiler lookup failed for device={device_id} "
                f"layer={layer_idx}: {e}. NO FALLBACK VALUES ALLOWED."
            ) from e

    def _boundary_comm_time(
        self,
        boundary_layer: int,
        left_group: list[int],
        right_group: list[int],
        model_config: ModelConfig,
        topology: DeviceTopology,
        profiler: ProfilerBackend | None,
    ) -> float:
        if not left_group or not right_group:
            return 0.0

        if profiler is not None:
            try:
                output_size = max(
                    profiler.get_output_size(boundary_layer),
                    1e-6,
                )
                # Use pairwise bandwidth between
                # representative devices of each group
                src_id = left_group[0]
                dst_id = right_group[0]
                pair_bw = profiler.get_pairwise_bandwidth(
                    src_id, dst_id,
                )
                if pair_bw <= 0:
                    raise RuntimeError(
                        f"Invalid pairwise bandwidth: "
                        f"{src_id}->{dst_id} = {pair_bw}"
                    )
                return output_size / pair_bw
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(
                    "Profiler comm lookup failed at boundary %s: %s",
                    boundary_layer, e,
                )
                # Fall through to topology lookup

        output_mb = self._default_output_mb(model_config)
        min_bw = float("inf")
        max_lat = 0.0
        for src in left_group:
            for dst in right_group:
                bw = self._lookup_metric_strict(topology.bandwidths, src, dst, "bandwidth")
                lat = self._lookup_metric_strict(topology.latencies, src, dst, "latency")
                min_bw = min(min_bw, max(bw, 1e-6))
                max_lat = max(max_lat, lat)
        return max_lat + output_mb / max(min_bw, 1e-6) * 1000.0

    def _default_output_mb(self, model_config: ModelConfig) -> float:
        """Activation payload in MB for inter-stage send/recv."""
        bs = max(1, model_config.micro_batch_size)
        bytes_per_element = 4.0
        total = bs * model_config.seq_length * model_config.embedding_dim
        return max(1e-6, total * bytes_per_element / (1024.0 * 1024.0))


__all__ = ["DTFMStrategy"]
