from __future__ import annotations

import logging
from typing_extensions import override

from baselines.core.config import (
    DeviceConfig,
    DeviceTopology,
    FaultToleranceConfig,
    ModelConfig,
    ParallelConfig,
    ParallelismPlan,
)
from baselines.core.profiler import ProfilerBackend

from .base import ParallelismStrategy

logger = logging.getLogger(__name__)


class ConfidentStrategy(ParallelismStrategy):
    """Confident baseline with DP bottleneck-minimizing stage partition."""

    def __init__(self, pp_size: int, dp_size: int) -> None:
        self.pp_size: int = max(1, pp_size)
        self.dp_size: int = max(1, dp_size)
        self.parallel_config: ParallelConfig = ParallelConfig(
            pp_size=self.pp_size,
            dp_size=self.dp_size,
            schedule_type=self.get_schedule_type(),
        )

    @override
    def create_plan(
        self,
        model_config: ModelConfig,
        device_topology: DeviceTopology,
        profiler: ProfilerBackend | None = None,
    ) -> ParallelismPlan:
        topology = self._normalize_topology(device_topology)
        num_layers = max(1, model_config.num_layers)
        # Use ALL available devices as pipeline stages
        # (matches notebook ConfidantScheduler: k = num_devices)
        num_devices = len(topology.device_specs)
        num_stages = min(num_devices, num_layers)
        num_stages = max(1, num_stages)
        device_groups = {
            stage_idx: [stage_idx] for stage_idx in range(num_stages)
        }

        partition_points, est_latency = self._dp_partition(
            model_config,
            topology,
            num_stages,
            profiler,
        )
        logger.info(
            "Confident strategy plan points=%s latency=%.2fms",
            partition_points,
            est_latency,
        )
        return ParallelismPlan(
            partition_points=partition_points,
            device_groups=device_groups,
            micro_batch_alloc={},
            schedule_type=self.get_schedule_type(),
            estimated_latency_ms=est_latency,
        )

    @override
    def get_schedule_type(self) -> str:
        return "1f1b"

    @override
    def get_fault_tolerance_config(self) -> FaultToleranceConfig:
        return FaultToleranceConfig(
            checkpoint_dir="./checkpoints/confident",
            checkpoint_interval=100,
            heartbeat_interval_s=0.0,
            heartbeat_timeout_s=0.0,
            backward_timeout_ms=30000.0,
            replication_mode="local_global",
            replication_interval=20,
            ft_check_interval=5,
        )

    def _normalize_topology(self, topology: DeviceTopology) -> DeviceTopology:
        specs = list(topology.device_specs)
        # Keep all devices — don't trim to pp_size
        target = max(len(specs), self.pp_size, 1)
        if not specs:
            specs = [DeviceConfig(device_id=i) for i in range(target)]
        elif len(specs) < target:
            start = len(specs)
            for idx in range(start, target):
                specs.append(DeviceConfig(device_id=idx))
        return DeviceTopology(
            device_specs=specs,
            bandwidths=dict(topology.bandwidths),
            latencies=dict(topology.latencies),
        )

    def _estimate_stage_memory_gb(
        self,
        model_config: ModelConfig,
        num_sublayers: int,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ) -> float:
        """Estimate memory usage in GB for a pipeline stage.

        Port of Confidant memory model with GQA support.
        Accounts for: weights, optimizer states, gradients, activations.
        """
        h = float(model_config.embedding_dim)
        v = float(model_config.vocab_size)
        # GQA: num_kv_heads may be less than num_attention_heads
        a = float(getattr(model_config, "num_attention_heads", h // 128))
        kv = float(getattr(model_config, "num_kv_heads", a))
        hff = float(getattr(model_config, "d_ff", 4 * h))
        s = float(model_config.seq_length)
        b = float(getattr(model_config, "micro_batch_size", 1))
        sublayer = float(num_sublayers)

        gqa_ratio = kv / a if a > 0 else 1.0
        bytes_to_gb = 2 ** -30

        # m1: Model weights (fp32 = 4 bytes)
        attn_params = 2 * h * h * (1 + gqa_ratio)
        ffn_params = 2 * h * hff
        transformer_params = attn_params + ffn_params + 12 * h

        m1 = (
            (h * v if is_first_stage else 0)
            + transformer_params * sublayer
            + (h * v if is_last_stage else 0)
        ) * 4 * bytes_to_gb

        # m2: Optimizer (AdamW) + gradients = 4× params
        m2 = transformer_params * sublayer * 4 * 4 * bytes_to_gb

        # m3: Activations
        attn_act = s * b * h * (2 + 2 * gqa_ratio)
        attn_scores = a * s * s * b
        ffn_act = 2 * s * b * hff
        m3 = sublayer * (attn_act + attn_scores + ffn_act) * bytes_to_gb

        return m1 + m2 + m3

    def _dp_partition(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        num_stages: int,
        profiler: ProfilerBackend | None,
    ) -> tuple[list[int], float]:
        num_layers = max(1, model_config.num_layers)
        if num_stages <= 1:
            return [], 0.0

        # Minimum layers per stage to avoid degenerate partitions
        # that cause pipeline bubble and NCCL timing issues
        MIN_LAYERS_PER_STAGE = max(2, num_layers // (num_stages * 2))
        logger.info(
            "DP partition: min_layers_per_stage=%d",
            MIN_LAYERS_PER_STAGE,
        )

        # Extract memory budgets from device specs (in GB)
        memory_budgets_gb = [
            spec.memory_budget_mb / 1024.0
            for spec in topology.device_specs[:num_stages]
        ]
        logger.info(
            "Memory budgets (GB): %s",
            [f"{m:.1f}" for m in memory_budgets_gb],
        )

        stage_prefix: list[list[float]] = []
        for stage_idx in range(num_stages):
            spec = topology.device_specs[stage_idx]
            cap = max(spec.compute_capacity, 0.1)
            prefix = [0.0]
            for layer_idx in range(num_layers):
                value = self._layer_time(
                    model_config,
                    spec,
                    stage_idx,
                    layer_idx,
                    profiler,
                )
                prefix.append(prefix[-1] + value / cap)
            stage_prefix.append(prefix)

        dp = [
            [float("inf")] * num_stages
            for _ in range(num_layers)
        ]
        split = [[-1] * num_stages for _ in range(num_layers)]

        def range_cost(stage_idx: int, start: int, end: int) -> float:
            prefix = stage_prefix[stage_idx]
            return prefix[end + 1] - prefix[start]

        # Base case: stage 0 (first stage, has embedding)
        # Must have at least MIN_LAYERS_PER_STAGE layers
        for end in range(MIN_LAYERS_PER_STAGE - 1, num_layers):
            num_sublayers = end + 1
            mem_est = self._estimate_stage_memory_gb(
                model_config,
                num_sublayers,
                is_first_stage=True,
                is_last_stage=(num_stages == 1),
            )
            if mem_est > memory_budgets_gb[0]:
                logger.debug(
                    "Stage 0: %d layers needs %.1fGB > budget %.1fGB",
                    num_sublayers, mem_est, memory_budgets_gb[0],
                )
                break  # Can't fit more layers on device 0
            dp[end][0] = range_cost(0, 0, end)

        # Fill DP table with memory constraints + min layers
        for stage_idx in range(1, num_stages):
            is_last = stage_idx == num_stages - 1
            mem_budget = memory_budgets_gb[stage_idx]
            # Minimum end index: need MIN_LAYERS for each stage up to here
            min_end = (stage_idx + 1) * MIN_LAYERS_PER_STAGE - 1
            for end in range(max(stage_idx, min_end), num_layers):
                # Minimum cut: previous stages need their min layers
                min_cut = stage_idx * MIN_LAYERS_PER_STAGE - 1
                # Maximum cut: this stage needs at least MIN_LAYERS
                max_cut = end - MIN_LAYERS_PER_STAGE
                for cut in range(max(stage_idx - 1, min_cut), min(end, max_cut + 1)):
                    # Layers cut+1..end on this stage
                    num_sublayers = end - cut
                    if num_sublayers < MIN_LAYERS_PER_STAGE:
                        continue  # Skip: too few layers
                    mem_est = self._estimate_stage_memory_gb(
                        model_config,
                        num_sublayers,
                        is_first_stage=False,
                        is_last_stage=is_last,
                    )
                    if mem_est > mem_budget:
                        continue  # Skip: exceeds memory budget

                    stage_time = range_cost(stage_idx, cut + 1, end)
                    comm = self._comm_time(
                        model_config,
                        topology,
                        cut,
                        stage_idx - 1,
                        stage_idx,
                        profiler,
                    )
                    candidate = max(dp[cut][stage_idx - 1], stage_time + comm)
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
            points.append(cut)
            end = cut
            stage_idx -= 1
        points.reverse()
        bottleneck = dp[num_layers - 1][num_stages - 1]
        if bottleneck == float("inf"):
            # Log what the memory model thinks each stage can handle
            for s in range(num_stages):
                max_layers = 0
                for n in range(1, num_layers + 1):
                    mem = self._estimate_stage_memory_gb(
                        model_config, n,
                        is_first_stage=(s == 0),
                        is_last_stage=(s == num_stages - 1),
                    )
                    if mem <= memory_budgets_gb[s]:
                        max_layers = n
                logger.error(
                    "Stage %d: budget=%.1fGB, max_layers=%d",
                    s, memory_budgets_gb[s], max_layers,
                )
            raise RuntimeError(
                f"DP partition failed: no valid partition found. "
                f"num_layers={num_layers} num_stages={num_stages}. "
                f"Check memory budgets vs model size. "
                f"NO FALLBACK VALUES ALLOWED."
            )
        return points, bottleneck

    def _layer_time(
        self,
        model_config: ModelConfig,
        spec: DeviceConfig,
        device_id: int,
        layer_idx: int,
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

    def _comm_time(
        self,
        model_config: ModelConfig,
        topology: DeviceTopology,
        boundary_layer: int,
        left_stage: int,
        right_stage: int,
        profiler: ProfilerBackend | None,
    ) -> float:
        """Compute communication time between stages.

        Uses profiler if available, otherwise falls back to topology.
        All bandwidth values MUST be measured via iperf3 - no defaults.
        """
        if profiler is not None:
            try:
                out_size = max(profiler.get_output_size(boundary_layer), 1e-6)
                left_bw = profiler.get_bandwidth(left_stage)
                right_bw = profiler.get_bandwidth(right_stage)
                if left_bw <= 0 or right_bw <= 0:
                    raise RuntimeError(
                        f"Invalid bandwidth: left={left_bw} right={right_bw}"
                    )
                return out_size / min(left_bw, right_bw)
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(
                    "Confident profiler comm miss boundary=%s: %s",
                    boundary_layer, e,
                )
                # Fall through to topology lookup

        left_id = topology.device_specs[left_stage].device_id
        right_id = topology.device_specs[right_stage].device_id
        bw = self._lookup_link_strict(topology.bandwidths, left_id, right_id, "bandwidth")
        lat = self._lookup_link_strict(topology.latencies, left_id, right_id, "latency")
        payload = model_config.seq_length * model_config.embedding_dim * 4.0
        payload_mb = payload / (1024.0 * 1024.0)
        return lat + payload_mb / max(bw, 1e-6) * 1000.0

    def _lookup_link_strict(
        self,
        table: dict[tuple[int, int], float],
        src: int,
        dst: int,
        metric_name: str,
    ) -> float:
        """Lookup link metric - raises error if not found.

        NO FALLBACK VALUES ALLOWED.
        """
        val = table.get((src, dst))
        if val is not None:
            return float(val)
        rev = table.get((dst, src))
        if rev is not None:
            return float(rev)
        raise RuntimeError(
            f"Missing {metric_name} for link ({src}, {dst}). "
            f"Re-run profiling with iperf3 on ALL nodes. "
            f"NO FALLBACK VALUES ALLOWED."
        )


__all__ = ["ConfidentStrategy"]
