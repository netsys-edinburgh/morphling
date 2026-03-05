from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

_LOGGER = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration for memory estimation.

    Matches the Confident Java implementation's config structure.
    """

    hidden_size: int = 4096  # h
    vocab_size: int = 50257  # v
    num_attention_heads: int = 32  # a
    intermediate_size: int = 11008  # hff (d_ff)
    max_seq_len: int = 512  # s
    batch_size: int = 2  # b (micro-batch size)
    num_layers: int = 32  # total transformer layers


class DPPartitioner:
    """DP-based layer partitioner that minimizes bottleneck stage time.

    Shared by DT-FM and Confident. Given per-layer execution times and
    inter-device bandwidths, finds the optimal partition of N layers
    across K pipeline stages.

    dp[i][j] = min bottleneck to assign layers 0..i across stages 0..j
    """

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        profiler_data: dict[str, object] | list[float] | None = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        self.num_layers = int(max(0, num_layers))
        self.num_devices = int(max(0, num_devices))
        self.profiler_data = profiler_data
        self.model_config = model_config

        self.time_intervals: dict[tuple[int, int, int, int], float] = {}
        self.output_sizes: list[float] = []
        self.bandwidths: list[float] = []
        self.computing_capacities: list[float] = []
        self.layer_times: list[float] | None = None

        if isinstance(profiler_data, Sequence) and not isinstance(
            profiler_data, dict
        ):
            self.layer_times = [float(v) for v in profiler_data]
        elif isinstance(profiler_data, dict):
            intervals_obj = profiler_data.get("time_intervals", {})
            if isinstance(intervals_obj, dict):
                parsed_intervals: dict[tuple[int, int, int, int], float] = {}
                for raw_key, raw_value in intervals_obj.items():
                    if not isinstance(raw_key, tuple) or len(raw_key) != 4:
                        continue
                    try:
                        key = (
                            int(raw_key[0]),
                            int(raw_key[1]),
                            int(raw_key[2]),
                            int(raw_key[3]),
                        )
                        parsed_intervals[key] = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                self.time_intervals = parsed_intervals

            self.output_sizes = self._to_float_list(
                profiler_data.get("output_sizes", [])
            )
            self.bandwidths = self._to_float_list(
                profiler_data.get("bandwidths", [])
            )
            self.computing_capacities = self._to_float_list(
                profiler_data.get("computing_capacities", [])
            )

            layer_times_obj = profiler_data.get("layer_times")
            if layer_times_obj is not None:
                self.layer_times = self._to_float_list(layer_times_obj)

        if self.num_layers == 0 and self.layer_times:
            self.num_layers = len(self.layer_times)

    @staticmethod
    def _to_float_list(value: object) -> list[float]:
        if isinstance(value, dict):
            return []
        if isinstance(value, (str, bytes, bytearray)):
            return []
        if not isinstance(value, Sequence):
            return []

        parsed: list[float] = []
        for item in value:
            if not isinstance(item, (int, float, str)):
                continue
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed

    def _effective_sizes(self) -> tuple[int, int]:
        n_layers = self.num_layers
        if n_layers <= 0 and self.layer_times:
            n_layers = len(self.layer_times)
        if n_layers <= 0:
            return 0, 0
        n_devices = max(1, min(self.num_devices, n_layers))
        return n_layers, n_devices

    def _capacity(self, device_id: int, is_average: bool) -> float:
        if not is_average:
            return 1.0
        if 0 <= device_id < len(self.computing_capacities):
            cap = self.computing_capacities[device_id]
            if cap > 0:
                return cap
        return 1.0

    def _get_time(self, device_id: int, start: int, end: int) -> float:
        if end < start:
            return 0.0

        if self.time_intervals:
            fwd = self.time_intervals.get((device_id, start, end, 0))
            bwd = self.time_intervals.get((device_id, start, end, 1))
            if fwd is not None and bwd is not None:
                return fwd + bwd

            total = 0.0
            found = False
            for layer_idx in range(start, end + 1):
                lf = self.time_intervals.get(
                    (device_id, layer_idx, layer_idx, 0)
                )
                lb = self.time_intervals.get(
                    (device_id, layer_idx, layer_idx, 1)
                )
                if lf is None and lb is None:
                    continue
                total += (lf or 0.0) + (lb or 0.0)
                found = True
            if found:
                return total

        if self.layer_times:
            lo = max(0, start)
            hi = min(end, len(self.layer_times) - 1)
            if lo <= hi:
                return sum(self.layer_times[lo : hi + 1])

        return float(max(0, end - start + 1))

    def _get_comm_time(self, layer_idx: int, device_id: int) -> float:
        if layer_idx < 0:
            return 0.0
        if not (0 <= layer_idx < len(self.output_sizes)):
            return 0.0
        if not (0 <= device_id < len(self.bandwidths)):
            return 0.0

        bandwidth = self.bandwidths[device_id]
        if bandwidth <= 0:
            return 0.0

        output_size_mb = self.output_sizes[layer_idx]
        return output_size_mb / bandwidth * 1000.0

    def _segment_memory_mb(self, start: int, end: int) -> float:
        if end < start:
            return 0.0
        if not self.output_sizes:
            return 0.0
        lo = max(0, start)
        hi = min(end, len(self.output_sizes) - 1)
        if lo > hi:
            return 0.0
        return float(sum(self.output_sizes[lo : hi + 1]))

    def _estimate_memory_gb(
        self,
        num_sublayers: int,
        device_idx: int,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ) -> float:
        """Estimate memory usage in GB for a pipeline stage.

        Port of Java DynamicScheduler.estimateMemory() from Confidant.
        Accounts for:
          m1: Model weights (embedding + transformer blocks + output head)
          m2: Optimizer states (AdamW: 4× for params + grads + m + v)
          m3: Activations (attention + FFN intermediates)
          m4: Optimizer replication for fault tolerance

        Args:
            num_sublayers: Number of transformer layers in this stage.
            device_idx: 0-indexed device position (for FT replication calc).
            is_first_stage: True if stage has embedding layer.
            is_last_stage: True if stage has output head (lm_head).

        Returns:
            Estimated memory in GB.
        """
        if self.model_config is None:
            # Fallback to simple heuristic if no config provided
            return num_sublayers * 0.5  # ~0.5 GB per layer estimate

        cfg = self.model_config
        h = float(cfg.hidden_size)
        v = float(cfg.vocab_size)
        a = float(cfg.num_attention_heads)
        hff = float(cfg.intermediate_size)
        s = float(cfg.max_seq_len)
        b = float(cfg.batch_size)
        sublayer = float(num_sublayers)

        # node = devices "before" this one (for FT replication)
        node = float(device_idx)

        # Embedding on first stage, output head on last stage
        embedding = 1.0 if is_first_stage else 0.0
        end_head = 1.0 if is_last_stage else 0.0

        # Bytes to GB: / 2^30
        bytes_to_gb = 2 ** -30

        # m1: Model weights (fp32 = 4 bytes)
        # - Embedding: h × v
        # - Transformer: (10h² + 12h) per layer
        #   10h² = Q,K,V,O projections (4 × h²) + FFN up/down (2 × h × hff ≈ 6h²)
        #   12h = biases and layer norms
        # - Output head: 2h (lm_head projection)
        m1 = (
            embedding * h * v
            + (10 * h * h + 12 * h) * sublayer
            + end_head * 2 * h
        ) * 4 * bytes_to_gb

        # m2: Optimizer (AdamW) + gradients
        # - params × 4 (param + grad + momentum + variance)
        m2 = (10 * h * h + 12 * h) * sublayer * 4 * 4 * bytes_to_gb

        # m3: Activations (mixed precision typical)
        # - 16 × s × b × h: attention Q,K,V,O outputs
        # - 8 × s × b × hff: FFN intermediates
        # - a × s² × b: attention scores (per head)
        m3 = (
            sublayer
            * (16 * s * b * h + 8 * s * b * hff + a * s * s * b)
            * bytes_to_gb
        )

        # m4: Fault tolerance - optimizer replication across nodes
        m4 = node * (10 * h * h + 12 * h) * sublayer * 4 * bytes_to_gb

        return m1 + m2 + m3 + m4

    def _equal_partition_points(
        self,
        n_layers: int,
        n_devices: int,
    ) -> list[int]:
        if n_devices <= 1:
            return []
        points: list[int] = []
        prev = -1
        for stage in range(1, n_devices):
            max_end = n_layers - (n_devices - stage) - 1
            end = (stage * n_layers) // n_devices - 1
            end = max(prev + 1, min(end, max_end))
            points.append(end)
            prev = end
        return points

    def partition(self, is_average: bool = True) -> list[int]:
        n_layers, n_devices = self._effective_sizes()
        if n_layers <= 0:
            return []
        if n_devices <= 1:
            return []

        inf = float("inf")
        dp = [[inf] * n_devices for _ in range(n_layers)]
        split = [[-1] * n_devices for _ in range(n_layers)]

        cap0 = self._capacity(0, is_average)
        for i in range(n_layers):
            dp[i][0] = self._get_time(0, 0, i) / cap0

        for j in range(1, n_devices):
            capj = self._capacity(j, is_average)
            for i in range(j, n_layers):
                for m in range(j - 1, i):
                    compute_t = self._get_time(j, m + 1, i) / capj
                    comm_t = self._get_comm_time(m, j - 1)
                    cost = max(dp[m][j - 1], compute_t + comm_t)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        points: list[int] = []
        i = n_layers - 1
        j = n_devices - 1
        while j > 0:
            m = split[i][j]
            if m < 0:
                _LOGGER.warning(
                    "DP backtrack failed. Falling back to equal partition."
                )
                return self._equal_partition_points(n_layers, n_devices)
            points.append(m)
            i = m
            j -= 1
        points.reverse()
        return points

    def partition_with_memory(
        self,
        memory_budgets: list[float],
        is_average: bool = True,
    ) -> list[int]:
        """DP partition with memory constraints.

        Port of Java calculatePartitionPointMemory() from Confidant.
        Uses _estimate_memory_gb() to check if a layer assignment fits
        within the device's memory budget.

        Args:
            memory_budgets: Memory budget in GB for each device.
            is_average: Whether to normalize by computing capacity.

        Returns:
            Partition points (layer indices ending each stage except last).
        """
        n_layers, n_devices = self._effective_sizes()
        if n_layers <= 0 or n_devices <= 1:
            return []
        if not memory_budgets:
            return self.partition(is_average=is_average)

        n_devices = min(n_devices, len(memory_budgets))
        inf = float("inf")
        dp = [[inf] * n_devices for _ in range(n_layers)]
        split = [[-1] * n_devices for _ in range(n_layers)]

        cap0 = self._capacity(0, is_average)
        # Base case: all layers 0..i on device 0 (first stage)
        for i in range(n_layers):
            num_sublayers = i + 1
            mem_est = self._estimate_memory_gb(
                num_sublayers,
                device_idx=0,
                is_first_stage=True,
                is_last_stage=(n_devices == 1),
            )
            if mem_est > memory_budgets[0]:
                break  # Can't fit more layers on device 0
            dp[i][0] = self._get_time(0, 0, i) / cap0

        # Fill DP table
        for j in range(1, n_devices):
            capj = self._capacity(j, is_average)
            is_last = j == n_devices - 1
            for i in range(j, n_layers):
                for m in range(j - 1, i):
                    # Layers m+1..i on device j
                    num_sublayers = i - m
                    mem_est = self._estimate_memory_gb(
                        num_sublayers,
                        device_idx=j,
                        is_first_stage=False,
                        is_last_stage=is_last,
                    )
                    if mem_est > memory_budgets[j]:
                        continue  # Skip: exceeds memory budget

                    compute_t = self._get_time(j, m + 1, i) / capj
                    comm_t = self._get_comm_time(m, j - 1)
                    cost = max(dp[m][j - 1], compute_t + comm_t)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        split[i][j] = m

        if dp[n_layers - 1][n_devices - 1] == inf:
            _LOGGER.warning(
                "No feasible memory-aware partition. Falling back to plain DP."
            )
            return self.partition(is_average=is_average)

        points: list[int] = []
        i = n_layers - 1
        j = n_devices - 1
        while j > 0:
            m = split[i][j]
            if m < 0:
                _LOGGER.warning(
                    "Memory-aware DP backtrack failed. Using plain DP."
                )
                return self.partition(is_average=is_average)
            points.append(m)
            i = m
            j -= 1
        points.reverse()
        return points


__all__ = ["DPPartitioner", "ModelConfig"]
