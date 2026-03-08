from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DeviceSpec:
    device_id: int
    memory_budget_mb: float = 80_000.0
    compute_capacity: float = 1.0


class AsteroidHPPPlanner:
    """Dynamic Programming HPP Planner (Asteroid Section 3.3).

    Finds optimal: model partition, device grouping, micro-batch allocation
    to minimize HPP-Round Latency under memory constraints.
    """

    def __init__(
        self,
        num_layers: int,
        num_devices: int,
        num_microbatches: int,
        micro_batch_size: int,
        profiler_data: dict[str, object] | None = None,
        device_specs: list[dict[str, float | int]] | None = None,
    ) -> None:
        self.num_layers = int(max(0, num_layers))
        self.num_devices = int(max(0, num_devices))
        self.num_microbatches = int(max(1, num_microbatches))
        self.micro_batch_size = int(max(1, micro_batch_size))

        data = profiler_data or {}
        self.exec_times = self._parse_exec_times(data.get("exec_times", {}))
        self.activation_sizes = self._to_float_list(
            data.get("activation_sizes", [])
        )
        self.weight_sizes = self._to_float_list(data.get("weight_sizes", []))
        # Boundary tensor sizes: the actual inter-stage
        # communication volume (output tensor only).
        # Falls back to activation_sizes for backwards compat.
        raw_boundary = data.get("boundary_sizes", [])
        parsed_boundary = self._to_float_list(raw_boundary)
        self.boundary_sizes = (
            parsed_boundary
            if parsed_boundary
            else list(self.activation_sizes)
        )
        self.bandwidths = self._parse_bandwidths(data.get("bandwidths", {}))
        self.default_bandwidth_mbps = self._infer_default_bandwidth()

        self.devices = self._prepare_devices(device_specs)
        self.devices, self.valid_splits = self._order_by_subnet(
            self.devices, self.bandwidths,
        )
        self.device_by_id = {
            device.device_id: device for device in self.devices
        }

    def _prepare_devices(
        self,
        specs: list[dict[str, float | int]] | None,
    ) -> list[_DeviceSpec]:
        devices: list[_DeviceSpec] = []
        if specs:
            for idx, spec in enumerate(specs):
                device_id = int(spec.get("device_id", idx))
                memory_budget_mb = float(spec.get("memory_budget_mb", 80_000.0))
                compute_capacity = float(spec.get("compute_capacity", 1.0))
                devices.append(
                    _DeviceSpec(
                        device_id=device_id,
                        memory_budget_mb=max(0.0, memory_budget_mb),
                        compute_capacity=max(1e-6, compute_capacity),
                    )
                )

        existing_ids = {device.device_id for device in devices}
        for device_id in range(self.num_devices):
            if device_id in existing_ids:
                continue
            devices.append(_DeviceSpec(device_id=device_id))

        devices.sort(key=lambda item: item.memory_budget_mb, reverse=True)
        return devices[: self.num_devices]

    def _infer_default_bandwidth(self) -> float:
        values = [bw for bw in self.bandwidths.values() if bw > 0]
        if not values:
            return 12_000.0
        return float(min(values))

    @staticmethod
    def _order_by_subnet(
        devices: list[_DeviceSpec],
        bandwidths: dict[tuple[int, int], float],
    ) -> tuple[list[_DeviceSpec], list[int]]:
        """Re-order devices so same-subnet peers are contiguous.

        Returns (ordered_devices, valid_splits) where valid_splits
        is a sorted list of device-count values at which a subnet
        boundary falls.  The DP must only split device groups at
        these counts so that every group consists of complete
        subnet clusters.

        Uses union-find: devices are merged when their mutual
        bandwidth exceeds 5× the minimum measured bandwidth.
        """
        N = len(devices)
        if N <= 2 or not bandwidths:
            return devices, list(range(N + 1))

        bws = [bw for bw in bandwidths.values() if bw > 0]
        if not bws:
            return devices, list(range(N + 1))
        threshold = min(bws) * 5

        ids = {d.device_id for d in devices}
        parent: dict[int, int] = {did: did for did in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for d1 in devices:
            for d2 in devices:
                if d1.device_id >= d2.device_id:
                    continue
                fwd = bandwidths.get(
                    (d1.device_id, d2.device_id), 0)
                rev = bandwidths.get(
                    (d2.device_id, d1.device_id), 0)
                if max(fwd, rev) >= threshold:
                    union(d1.device_id, d2.device_id)

        clusters: dict[int, list[_DeviceSpec]] = defaultdict(list)
        for d in devices:
            clusters[find(d.device_id)].append(d)

        ordered: list[_DeviceSpec] = []
        for cluster in sorted(
            clusters.values(),
            key=lambda c: min(d.device_id for d in c),
        ):
            ordered.extend(
                sorted(cluster, key=lambda d: d.device_id))

        # Valid split counts: device groups must be whole
        # subnet clusters.  We record cumulative sizes
        # from the RIGHT of the ordered list (because the
        # DP builds groups from the right end).
        cluster_sizes = [
            len(c) for c in sorted(
                clusters.values(),
                key=lambda c: min(d.device_id for d in c),
            )
        ]
        # Cumulative from right
        valid_splits = {0}
        cumul = 0
        for sz in reversed(cluster_sizes):
            cumul += sz
            valid_splits.add(cumul)
        valid_splits = sorted(valid_splits)

        _LOGGER.info(
            "Device order (subnet-clustered): %s  "
            "valid_splits=%s",
            [d.device_id for d in ordered],
            valid_splits,
        )
        return ordered, valid_splits

    def _device_or_default(self, device_id: int) -> _DeviceSpec:
        return self.device_by_id.get(
            device_id,
            _DeviceSpec(device_id=device_id),
        )

    @staticmethod
    def _to_float_list(value: object) -> list[float]:
        if isinstance(value, dict):
            return []
        if isinstance(value, (str, bytes, bytearray)):
            return []
        if not isinstance(value, list):
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

    @staticmethod
    def _parse_bandwidths(value: object) -> dict[tuple[int, int], float]:
        parsed: dict[tuple[int, int], float] = {}
        if not isinstance(value, dict):
            return parsed

        for key, raw_bw in value.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            try:
                src = int(key[0])
                dst = int(key[1])
                parsed[(src, dst)] = float(raw_bw)
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def _parse_exec_times(
        value: object,
    ) -> dict[int, dict[int, dict[int, tuple[float, float]]]]:
        parsed: dict[int, dict[int, dict[int, tuple[float, float]]]] = {}
        if not isinstance(value, dict):
            return parsed

        for raw_dev, raw_layers in value.items():
            if not isinstance(raw_layers, dict):
                continue
            try:
                dev_id = int(raw_dev)
            except (TypeError, ValueError):
                continue

            layer_map: dict[int, dict[int, tuple[float, float]]] = {}
            for raw_layer, raw_bs_map in raw_layers.items():
                if not isinstance(raw_bs_map, dict):
                    continue
                try:
                    layer_idx = int(raw_layer)
                except (TypeError, ValueError):
                    continue

                bs_map: dict[int, tuple[float, float]] = {}
                for raw_bs, raw_pair in raw_bs_map.items():
                    if not isinstance(raw_pair, (list, tuple)):
                        continue
                    if len(raw_pair) != 2:
                        continue
                    try:
                        batch_size = int(raw_bs)
                        fwd_ms = float(raw_pair[0])
                        bwd_ms = float(raw_pair[1])
                    except (TypeError, ValueError):
                        continue
                    bs_map[batch_size] = (fwd_ms, bwd_ms)

                if bs_map:
                    layer_map[layer_idx] = bs_map

            if layer_map:
                parsed[dev_id] = layer_map

        return parsed

    def _get_exec_time(
        self,
        device_id: int,
        layer_idx: int,
        batch_size: int,
    ) -> tuple[float, float]:
        times = self.exec_times.get(device_id, {}).get(layer_idx, {})
        if not times:
            return 1.0, 2.0

        normalized = {int(k): v for k, v in times.items()}
        if batch_size in normalized:
            fwd_ms, bwd_ms = normalized[batch_size]
            return float(fwd_ms), float(bwd_ms)

        keys = sorted(normalized.keys())
        if batch_size <= keys[0]:
            fwd_ms, bwd_ms = normalized[keys[0]]
            return float(fwd_ms), float(bwd_ms)

        if batch_size >= keys[-1]:
            ref_bs = keys[-1]
            ref_fwd, ref_bwd = normalized[ref_bs]
            ratio = (batch_size / ref_bs) ** 0.85
            return float(ref_fwd) * ratio, float(ref_bwd) * ratio

        for idx in range(len(keys) - 1):
            lo = keys[idx]
            hi = keys[idx + 1]
            if lo <= batch_size <= hi:
                alpha = (batch_size - lo) / (hi - lo)
                lo_fwd, lo_bwd = normalized[lo]
                hi_fwd, hi_bwd = normalized[hi]
                fwd_ms = float(lo_fwd) * (1 - alpha) + float(hi_fwd) * alpha
                bwd_ms = float(lo_bwd) * (1 - alpha) + float(hi_bwd) * alpha
                return fwd_ms, bwd_ms

        return 1.0, 2.0

    def _memory_footprint(
        self,
        stage_idx: int,
        num_stages: int,
        start_l: int,
        end_l: int,
        batch_size: int,
    ) -> float:
        k_p = max(1, 2 * (num_stages - stage_idx) - 1)

        weight_bytes = sum(
            self.weight_sizes[layer_idx]
            for layer_idx in range(start_l, end_l)
            if layer_idx < len(self.weight_sizes)
        )
        mem_mod = weight_bytes
        mem_grad = weight_bytes          # gradient memory
        mem_opt = weight_bytes * 2        # AdamW momentum + variance

        mem_act = (
            sum(
                self.activation_sizes[layer_idx]
                for layer_idx in range(start_l, end_l)
                if layer_idx < len(self.activation_sizes)
            )
            * batch_size
        )
        # 1.10x safety factor for PyTorch overhead,
        # CUDA context, and memory fragmentation
        total_bytes = (mem_mod + mem_grad + mem_opt + k_p * mem_act)
        return total_bytes / (1024 * 1024) * 1.10

    def _compute_device_capacity(
        self,
        device_id: int,
        start_l: int,
        end_l: int,
        batch_size: int,
    ) -> float:
        """Eq. (9): v_d = (Σ [t_f(B) + t_b(B)])^{-1}."""
        total = 0.0
        for layer_idx in range(start_l, end_l):
            fwd_ms, bwd_ms = self._get_exec_time(
                device_id, layer_idx, batch_size,
            )
            total += fwd_ms + bwd_ms
        return 1.0 / max(total, 1e-9)

    def _alloc_microbatch(
        self,
        device_ids: list[int],
        start_l: int,
        end_l: int,
        micro_bs: int,
    ) -> tuple[dict[int, int], float, float]:
        if not device_ids:
            return {}, float("inf"), float("inf")

        devices_here = [self._device_or_default(did) for did in device_ids]
        alloc = {device.device_id: 0 for device in devices_here}
        remaining = micro_bs
        active = list(devices_here)

        # Eq. (9): compute v_d from profiled execution times
        capacities: dict[int, float] = {}
        for device in devices_here:
            capacities[device.device_id] = (
                self._compute_device_capacity(
                    device.device_id,
                    start_l,
                    end_l,
                    micro_bs,
                )
            )

        while remaining > 0 and active:
            total_cap = sum(
                capacities[device.device_id]
                for device in active
            )
            if total_cap <= 0:
                break

            new_active: list[_DeviceSpec] = []
            for device in active:
                cap = capacities[device.device_id]
                share = max(
                    1,
                    int(round(cap / total_cap * remaining)),
                )
                mem_needed = self._memory_footprint(
                    0,
                    1,
                    start_l,
                    end_l,
                    share,
                )
                max_bs = share
                while mem_needed > device.memory_budget_mb and max_bs > 1:
                    max_bs -= 1
                    mem_needed = self._memory_footprint(
                        0,
                        1,
                        start_l,
                        end_l,
                        max_bs,
                    )

                actual = min(share, max_bs, remaining)
                alloc[device.device_id] += actual
                remaining -= actual

                if mem_needed < device.memory_budget_mb * 0.95:
                    new_active.append(device)

            active = new_active

        if remaining > 0:
            ordered = sorted(
                devices_here,
                key=lambda dev: capacities[dev.device_id],
                reverse=True,
            )
            progress = True
            while remaining > 0 and progress:
                progress = False
                for device in ordered:
                    candidate = alloc[device.device_id] + 1
                    mem_needed = self._memory_footprint(
                        0,
                        1,
                        start_l,
                        end_l,
                        candidate,
                    )
                    if mem_needed <= device.memory_budget_mb:
                        alloc[device.device_id] = candidate
                        remaining -= 1
                        progress = True
                        if remaining == 0:
                            break

        def exec_time(device_id: int, bs: int) -> float:
            if bs <= 0:
                return 0.0

            total = 0.0
            for layer_idx in range(start_l, end_l):
                fwd_ms, bwd_ms = self._get_exec_time(device_id, layer_idx, bs)
                total += fwd_ms + bwd_ms
            return total

        # Alg. 1 Phase 2: straggler offloading until convergence
        while True:
            times: dict[int, float] = {
                device_id: exec_time(device_id, bs)
                for device_id, bs in alloc.items()
                if bs > 0
            }
            if not times:
                break

            slowest = max(
                times, key=lambda device_id: times[device_id],
            )
            fastest = min(
                times, key=lambda device_id: times[device_id],
            )
            if slowest == fastest or alloc[slowest] <= 1:
                break

            old_time = times[slowest]
            alloc[slowest] -= 1
            alloc[fastest] += 1
            new_time = max(
                exec_time(slowest, alloc[slowest]),
                exec_time(fastest, alloc[fastest]),
            )
            if new_time >= old_time:
                alloc[slowest] += 1
                alloc[fastest] -= 1
                break

        def exec_time_split(
            device_id: int,
            bs: int,
        ) -> tuple[float, float]:
            if bs <= 0:
                return 0.0, 0.0
            total_fwd = 0.0
            total_bwd = 0.0
            for layer_idx in range(start_l, end_l):
                f, b = self._get_exec_time(
                    device_id, layer_idx, bs,
                )
                total_fwd += f
                total_bwd += b
            return total_fwd, total_bwd

        if any(bs > 0 for bs in alloc.values()):
            # Eq. (8): E_f = max_d Σ t_f, E_b = max_d Σ t_b
            straggler_fwd = max(
                exec_time_split(did, bs)[0]
                for did, bs in alloc.items()
            )
            straggler_bwd = max(
                exec_time_split(did, bs)[1]
                for did, bs in alloc.items()
            )
        else:
            straggler_fwd = float("inf")
            straggler_bwd = float("inf")

        return alloc, straggler_fwd, straggler_bwd

    def _comm_time_inter_stage(
        self,
        layer_idx: int,
        src_group: list[int],
        dst_group: list[int],
        batch_size: int,
    ) -> tuple[float, float]:
        """Inter-stage comm split into (forward, backward).

        Forward = activations src->dst,
        Backward = gradients dst->src.
        """
        if not src_group or not dst_group:
            return 0.0, 0.0
        sizes = (
            self.boundary_sizes
            if self.boundary_sizes
            else self.activation_sizes
        )
        if not sizes:
            return 0.0, 0.0

        bs = batch_size if batch_size > 0 else self.micro_batch_size
        idx = min(max(layer_idx, 0), len(sizes) - 1)
        act_size = sizes[idx] * bs
        act_size_mb = act_size / (1024 * 1024)

        min_bw = float("inf")
        for src in src_group:
            for dst in dst_group:
                bw = self.bandwidths.get(
                    (src, dst),
                    self.default_bandwidth_mbps,
                )
                min_bw = min(min_bw, bw)

        if min_bw <= 0 or min_bw == float("inf"):
            return float("inf"), float("inf")

        comm_per_dir = act_size_mb / min_bw * 1000
        return comm_per_dir, comm_per_dir

    def _allreduce_time(
        self,
        device_group: list[int],
        start_l: int,
        end_l: int,
    ) -> float:
        g_size = len(device_group)
        if g_size <= 1:
            return 0.0

        weight_bytes = sum(
            self.weight_sizes[layer_idx]
            for layer_idx in range(start_l, end_l)
            if layer_idx < len(self.weight_sizes)
        )

        min_bw = float("inf")
        for d1 in device_group:
            for d2 in device_group:
                if d1 == d2:
                    continue
                bw = self.bandwidths.get((d1, d2), self.default_bandwidth_mbps)
                min_bw = min(min_bw, bw)

        if min_bw <= 0 or min_bw == float("inf"):
            return float("inf")

        vol_mb = 2 * (g_size - 1) / g_size * weight_bytes / (1024 * 1024)
        return vol_mb / min_bw * 1000

    def _determine_dominant_step(
        self,
        step_ef: list[float],
        step_eb: list[float],
    ) -> int:
        """Eq. (11): find the dominant step.

        For step c the aligned execution time is
        aligned(c) = M*(E_f^c + E_b^c) + Σ_{i<c}(E_f^i + E_b^i).
        The step with the largest aligned time has the fewest
        bubbles and is therefore the dominant step.
        """
        M = self.num_microbatches
        best_idx = 0
        best_aligned = 0.0
        prefix_sum = 0.0
        for c in range(len(step_ef)):
            aligned = (
                M * (step_ef[c] + step_eb[c]) + prefix_sum
            )
            if aligned > best_aligned:
                best_aligned = aligned
                best_idx = c
            prefix_sum += step_ef[c] + step_eb[c]
        return best_idx

    def _compute_hpp_latency(
        self,
        step_ef: list[float],
        step_eb: list[float],
        step_ar: list[float],
        dominant_step: int,
    ) -> float:
        """Eqs. (4)-(6): HPP-Round Latency.

        T_w^s  = Σ_{i<s} E_f^i               (Eq. 5)
        T_e^s  = M*(E_f^dm+E_b^dm)+off(s,dm)  (Eq. 6)
        T_a^s  = AllReduce for step s           (Eq. 5)
        Latency = max_s(T_w^s + T_e^s + T_a^s) (Eq. 4)
        """
        S = len(step_ef)
        if S == 0:
            return float("inf")
        M = self.num_microbatches
        dm = dominant_step
        dm_exec = step_ef[dm] + step_eb[dm]
        max_lat = 0.0
        for s in range(S):
            # T_w^s: sum of forward times before step s
            Tw = sum(step_ef[i] for i in range(s))
            # T_e^s via Eq. (6)
            if s < dm:
                offset = sum(
                    step_ef[i] + step_eb[i]
                    for i in range(s, dm)
                )
                Te = M * dm_exec + offset
            elif s == dm:
                Te = M * dm_exec
            else:
                offset = sum(
                    step_ef[i] + step_eb[i]
                    for i in range(dm, s)
                )
                Te = max(0.0, M * dm_exec - offset)
            # T_a^s
            Ta = step_ar[s]
            total = Tw + Te + Ta
            max_lat = max(max_lat, total)
        return max_lat

    def plan(self) -> dict[str, object]:
        l_total = self.num_layers
        n_total = min(self.num_devices, len(self.devices))
        if l_total <= 0 or n_total <= 0:
            raise RuntimeError(
                f"Invalid configuration: num_layers={l_total} "
                f"num_devices={n_total}. Both must be > 0. "
                f"NO FALLBACK PLANS ALLOWED."
            )

        device_ids = [device.device_id for device in self.devices[:n_total]]
        best_plan: dict[str, object] | None = None
        best_latency = float("inf")

        max_stages = min(l_total, n_total)
        for p in range(1, max_stages + 1):
            result = self._dp_plan(l_total, n_total, p, device_ids)
            if result is None:
                continue

            latency_obj = result.get("latency")
            if not isinstance(latency_obj, (int, float)):
                continue
            latency = float(latency_obj)
            if latency < best_latency:
                best_latency = latency
                best_plan = result

        if best_plan is None:
            raise RuntimeError(
                f"DP planning failed: no valid plan found for "
                f"num_layers={l_total} num_devices={n_total}. "
                f"Ensure all profiling data (layer timings, bandwidth) "
                f"is available. NO FALLBACK PLANS ALLOWED."
            )

        _LOGGER.info(
            "Asteroid planner: stages=%s partition=%s latency=%.3fms",
            best_plan["num_stages"],
            best_plan["partition_points"],
            best_plan["latency"],
        )
        return best_plan

    def _normalize_partition_points(self, points: list[int]) -> list[int]:
        normalized = sorted(set(int(point) for point in points))
        return [point for point in normalized if 0 < point < self.num_layers]

    def _dp_plan(
        self,
        L: int,
        N: int,
        P: int,
        device_ids: list[int],
    ) -> dict[str, object] | None:
        if P > L or P > N:
            return None

        inf = float("inf")
        q = [
            [[inf for _ in range(P + 1)] for _ in range(N + 1)]
            for _ in range(L + 1)
        ]
        configs: list[list[list[dict[str, object] | None]]] = [
            [[None for _ in range(P + 1)] for _ in range(N + 1)]
            for _ in range(L + 1)
        ]

        # Only consider device counts at subnet boundaries
        # so each group is a complete subnet cluster.
        vs = self.valid_splits

        for l in range(1, L + 1):
            for n in vs:
                if n < 1 or n > N:
                    continue
                group = device_ids[N - n :]
                alloc, exec_fwd, exec_bwd = (
                    self._alloc_microbatch(
                        group,
                        L - l,
                        L,
                        self.micro_batch_size,
                    )
                )

                # Memory check for single-stage base case
                mem_ok = True
                # When used as the last stage in P stages,
                # stage_idx = P - 1
                global_stage_idx = max(0, P - 1)
                for did, bs in alloc.items():
                    if bs <= 0:
                        continue
                    mem = self._memory_footprint(
                        global_stage_idx,
                        P,
                        L - l,
                        L,
                        bs,
                    )
                    device = self._device_or_default(did)
                    if mem > device.memory_budget_mb:
                        mem_ok = False
                        break
                if not mem_ok:
                    continue

                ar_t = self._allreduce_time(group, L - l, L)
                # Single execution step (Alg. 2 base case)
                s_ef = [exec_fwd]
                s_eb = [exec_bwd]
                s_ar = [ar_t]
                dm = 0
                lat = self._compute_hpp_latency(
                    s_ef, s_eb, s_ar, dm,
                )
                if lat < q[l][n][1]:
                    q[l][n][1] = lat
                    configs[l][n][1] = {
                        "partition": [L - l],
                        "groups": {0: group},
                        "allocs": {0: alloc},
                        "step_ef": s_ef,
                        "step_eb": s_eb,
                        "step_ar": s_ar,
                        "dominant_step": dm,
                    }

        for p in range(2, P + 1):
            for l in range(p, L + 1):
                for n in vs:
                    if n < p or n > N:
                        continue
                    for l_prime in range(p - 1, l):
                        for n_prime in vs:
                            if n_prime < (p - 1) or n_prime >= n:
                                continue
                            if q[l_prime][n_prime][p - 1] >= inf:
                                continue

                            prev_cfg = configs[l_prime][n_prime][p - 1]
                            if prev_cfg is None:
                                continue

                            new_group = device_ids[N - n : N - n_prime]
                            if not new_group:
                                continue

                            start_l = L - l
                            end_l = L - l_prime
                            alloc, exec_fwd, exec_bwd = (
                                self._alloc_microbatch(
                                    new_group,
                                    start_l,
                                    end_l,
                                    self.micro_batch_size,
                                )
                            )

                            global_stage_idx = P - p
                            mem_ok = True
                            for did, bs in alloc.items():
                                if bs <= 0:
                                    continue
                                mem = self._memory_footprint(
                                    global_stage_idx,
                                    P,
                                    start_l,
                                    end_l,
                                    bs,
                                )
                                device = self._device_or_default(did)
                                if mem > device.memory_budget_mb:
                                    mem_ok = False
                                    break
                            if not mem_ok:
                                continue

                            groups_obj = prev_cfg.get("groups", {})
                            if not isinstance(groups_obj, dict):
                                continue

                            # Stage 0 in prev_cfg is the leftmost
                            # (immediately downstream) stage —
                            # the one the new stage sends
                            # activations to.
                            prev_last_group_obj = (
                                groups_obj.get(0, [])
                            )
                            if isinstance(
                                prev_last_group_obj, list,
                            ):
                                prev_last_group = [
                                    int(device_id)
                                    for device_id
                                    in prev_last_group_obj
                                ]
                            else:
                                prev_last_group = []

                            total_alloc_bs = sum(alloc.values())
                            if prev_last_group:
                                comm_fwd, comm_bwd = (
                                    self._comm_time_inter_stage(
                                        end_l - 1,
                                        new_group,
                                        prev_last_group,
                                        batch_size=total_alloc_bs,
                                    )
                                )
                            else:
                                comm_fwd, comm_bwd = 0.0, 0.0

                            ar_t = self._allreduce_time(
                                new_group,
                                start_l,
                                end_l,
                            )

                            # Build step lists (Alg. 2 §6-9)
                            old_ef = prev_cfg.get(
                                "step_ef", [],
                            )
                            old_eb = prev_cfg.get(
                                "step_eb", [],
                            )
                            old_ar = prev_cfg.get(
                                "step_ar", [],
                            )
                            new_step_ef = (
                                [exec_fwd, comm_fwd]
                                + list(old_ef)
                            )
                            new_step_eb = (
                                [exec_bwd, comm_bwd]
                                + list(old_eb)
                            )
                            new_step_ar = (
                                [ar_t, 0.0]
                                + list(old_ar)
                            )

                            # Eq. (11): dominant step
                            new_dm = (
                                self._determine_dominant_step(
                                    new_step_ef,
                                    new_step_eb,
                                )
                            )

                            # Eqs. (4)-(6): HPP-Round Latency
                            total_lat = (
                                self._compute_hpp_latency(
                                    new_step_ef,
                                    new_step_eb,
                                    new_step_ar,
                                    new_dm,
                                )
                            )

                            if total_lat < q[l][n][p]:
                                prev_groups = (
                                    copy.deepcopy(groups_obj)
                                )
                                allocs_obj = prev_cfg.get(
                                    "allocs", {},
                                )
                                if not isinstance(
                                    allocs_obj, dict,
                                ):
                                    continue
                                prev_allocs = (
                                    copy.deepcopy(allocs_obj)
                                )
                                groups_new = {
                                    0: list(new_group),
                                }
                                allocs_new = {
                                    0: dict(alloc),
                                }
                                for (
                                    stage_idx,
                                    group,
                                ) in prev_groups.items():
                                    if not isinstance(
                                        group, list,
                                    ):
                                        continue
                                    groups_new[
                                        int(stage_idx) + 1
                                    ] = [
                                        int(device_id)
                                        for device_id
                                        in group
                                    ]
                                for (
                                    stage_idx,
                                    stage_alloc,
                                ) in prev_allocs.items():
                                    if not isinstance(
                                        stage_alloc, dict,
                                    ):
                                        continue
                                    allocs_new[
                                        int(stage_idx) + 1
                                    ] = dict(stage_alloc)

                                partition_obj = (
                                    prev_cfg.get(
                                        "partition", [],
                                    )
                                )
                                if isinstance(
                                    partition_obj, list,
                                ):
                                    partition_new = [
                                        start_l,
                                    ] + [
                                        int(point)
                                        for point
                                        in partition_obj
                                    ]
                                else:
                                    partition_new = [start_l]

                                q[l][n][p] = total_lat
                                configs[l][n][p] = {
                                    "partition": partition_new,
                                    "groups": groups_new,
                                    "allocs": allocs_new,
                                    "step_ef": new_step_ef,
                                    "step_eb": new_step_eb,
                                    "step_ar": new_step_ar,
                                    "dominant_step": new_dm,
                                }

        best_lat = q[L][N][P]
        best_cfg = configs[L][N][P]
        if best_lat >= inf or best_cfg is None:
            return None

        groups = best_cfg.get("groups", {})
        if not isinstance(groups, dict):
            groups = {}

        allocs = best_cfg.get("allocs", {})
        if not isinstance(allocs, dict):
            allocs = {}

        partition_obj = best_cfg.get("partition", [])
        if isinstance(partition_obj, list):
            partition_raw = [int(point) for point in partition_obj]
        else:
            partition_raw = []

        partition_points = self._normalize_partition_points(partition_raw)
        return {
            "num_stages": P,
            "partition_points": partition_points,
            "device_groups": groups,
            "micro_batch_alloc": allocs,
            "latency": float(best_lat),
        }


__all__ = ["AsteroidHPPPlanner"]
