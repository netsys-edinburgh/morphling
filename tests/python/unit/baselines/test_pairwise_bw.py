"""Tests for pairwise bandwidth in profiler and strategies.

Verifies that _comm_time / _boundary_comm_time use the
actual iperf3-measured link between adjacent stages, not
each rank's worst-case link to any peer.
"""
from __future__ import annotations

import pytest

from baselines.core.profiler import ProfilerBackend


# ── Minimal profiler stubs ──────────────────────────────


class _MinBWProfiler(ProfilerBackend):
    """Old-style profiler: per-rank min bandwidth only."""

    def __init__(
        self,
        per_rank_bw: dict[int, float],
    ) -> None:
        self._per_rank_bw = per_rank_bw

    def profile_layer(self, m, d, n):
        return (1.0, 2.0, 0.0)

    def profile_bandwidth(self, s, d, sz):
        return 100.0

    def get_memory_info(self, dev):
        return (40960.0, 40960.0)

    def get_time_interval(self, dev, s, e, p):
        return 1.0

    def get_output_size(self, layer_idx):
        return 8.0  # MB

    def get_bandwidth(self, device_id):
        return self._per_rank_bw[device_id]

    def get_computing_capacity(self, device_id):
        return 1.0

    def get_available_memory(self, device_id):
        return 40960.0


class _PairBWProfiler(_MinBWProfiler):
    """New profiler: overrides get_pairwise_bandwidth
    with actual link data."""

    def __init__(
        self,
        per_rank_bw: dict[int, float],
        pair_bw: dict[tuple[int, int], float],
    ) -> None:
        super().__init__(per_rank_bw)
        self._pair_bw = pair_bw

    def get_pairwise_bandwidth(
        self, src_id: int, dst_id: int,
    ) -> float:
        pair = self._pair_bw.get((src_id, dst_id))
        if pair is not None:
            return pair
        pair = self._pair_bw.get((dst_id, src_id))
        if pair is not None:
            return pair
        return min(
            self.get_bandwidth(src_id),
            self.get_bandwidth(dst_id),
        )


# ── Base interface tests ────────────────────────────────


class TestProfilerPairwiseFallback:
    """Default get_pairwise_bandwidth falls back to
    min(get_bandwidth(src), get_bandwidth(dst))."""

    def test_fallback_uses_min(self) -> None:
        p = _MinBWProfiler({0: 0.5, 1: 1.0, 2: 0.1})
        # No override → should use min
        assert p.get_pairwise_bandwidth(0, 1) == 0.5
        assert p.get_pairwise_bandwidth(1, 2) == 0.1
        assert p.get_pairwise_bandwidth(0, 2) == 0.1

    def test_override_uses_pair(self) -> None:
        p = _PairBWProfiler(
            per_rank_bw={0: 0.04, 1: 0.04, 2: 0.12},
            pair_bw={(0, 1): 1.43, (1, 2): 0.04},
        )
        # (0,1) has direct measurement = 1.43
        assert p.get_pairwise_bandwidth(0, 1) == 1.43
        # Reverse direction should also work
        assert p.get_pairwise_bandwidth(1, 0) == 1.43
        # (1,2) has direct measurement = 0.04
        assert p.get_pairwise_bandwidth(1, 2) == 0.04
        # (0,2) not profiled → falls back to min
        assert p.get_pairwise_bandwidth(0, 2) == 0.04


# ── Adapter tests ────────────────────────────────────────


class TestProfileDataAdapterPairwise:
    """_ProfileDataAdapter.get_pairwise_bandwidth uses
    actual iperf3 pair, not per-rank min."""

    def _make_adapter(self):
        """Build adapter with realistic asymmetric BW."""
        # Simulate: R0↔R1 fast (11 Gbps), R1↔R2 slow (340 Mbps)
        from baselines.scripts.run_asteroid_planner import (
            _ProfileDataAdapter,
        )

        profiles = {
            0: {
                "layer_profile": {
                    "1": {"latency_ms_mean": 14.0}
                },
                "network": {
                    "10.0.0.2:5201": {
                        "ok": True,
                        "bandwidth_mbps": 11440.0,
                    },
                    "10.0.0.3:5201": {
                        "ok": True,
                        "bandwidth_mbps": 341.0,
                    },
                },
            },
            1: {
                "layer_profile": {
                    "1": {"latency_ms_mean": 14.0}
                },
                "network": {
                    "10.0.0.1:5201": {
                        "ok": True,
                        "bandwidth_mbps": 18640.0,
                    },
                    "10.0.0.3:5201": {
                        "ok": True,
                        "bandwidth_mbps": 339.0,
                    },
                },
            },
            2: {
                "layer_profile": {
                    "1": {"latency_ms_mean": 22.0}
                },
                "network": {
                    "10.0.0.1:5201": {
                        "ok": True,
                        "bandwidth_mbps": 1062.0,
                    },
                    "10.0.0.2:5201": {
                        "ok": True,
                        "bandwidth_mbps": 1022.0,
                    },
                },
            },
        }
        cluster_nodes = {
            "10.0.0.1": {"rank": 0},
            "10.0.0.2": {"rank": 1},
            "10.0.0.3": {"rank": 2},
        }
        return _ProfileDataAdapter(
            profiles=profiles,
            cluster_nodes=cluster_nodes,
            num_layers=8,
        )

    def test_fast_link_returns_fast_bw(self) -> None:
        adapter = self._make_adapter()
        # R0→R1: 11440 Mbps → 11440/8000 = 1.43 MB/ms
        bw = adapter.get_pairwise_bandwidth(0, 1)
        assert bw == pytest.approx(11440.0 / 8000.0, rel=1e-3)

    def test_slow_link_returns_slow_bw(self) -> None:
        adapter = self._make_adapter()
        # R1→R2: 339 Mbps → 339/8000 = 0.042375 MB/ms
        bw = adapter.get_pairwise_bandwidth(1, 2)
        assert bw == pytest.approx(339.0 / 8000.0, rel=1e-3)

    def test_reverse_direction(self) -> None:
        adapter = self._make_adapter()
        # R1→R0: 18640 Mbps from R1's profile
        bw_fw = adapter.get_pairwise_bandwidth(0, 1)
        bw_rv = adapter.get_pairwise_bandwidth(1, 0)
        # Forward (0→1) uses _bw_pairs[(0,1)] = 11440
        assert bw_fw == pytest.approx(11440.0 / 8000.0, rel=1e-3)
        # Reverse (1→0) uses _bw_pairs[(1,0)] = 18640
        assert bw_rv == pytest.approx(18640.0 / 8000.0, rel=1e-3)

    def test_old_get_bandwidth_still_returns_min(self) -> None:
        adapter = self._make_adapter()
        # R0's min across peers: min(11440, 341) = 341
        bw = adapter.get_bandwidth(0)
        assert bw == pytest.approx(341.0 / 8000.0, rel=1e-3)
        # R1's min across peers: min(18640, 339) = 339
        bw = adapter.get_bandwidth(1)
        assert bw == pytest.approx(339.0 / 8000.0, rel=1e-3)

    def test_pairwise_differs_from_min(self) -> None:
        """The whole point: pairwise R0→R1 should be much
        faster than the old min-based approach."""
        adapter = self._make_adapter()
        old_bw = min(
            adapter.get_bandwidth(0),
            adapter.get_bandwidth(1),
        )
        new_bw = adapter.get_pairwise_bandwidth(0, 1)
        # Old: min(341, 339)/8000 ≈ 0.042
        # New: 11440/8000 ≈ 1.43
        # Pairwise should be ~33x faster
        assert new_bw > old_bw * 10


# ── Comm time impact test ────────────────────────────────


class TestCommTimePairwise:
    """Verify strategies use pairwise BW for comm cost."""

    def test_confident_uses_pairwise(self) -> None:
        """Confident _comm_time on a fast link should be
        much cheaper than using per-rank min."""
        from baselines.core.config import (
            DeviceConfig,
            DeviceTopology,
            ModelConfig,
        )
        from baselines.strategies.confident_strategy import (
            ConfidentStrategy,
        )

        model_config = ModelConfig(
            model_type="opt",
            num_layers=8,
            embedding_dim=2048,
            num_heads=32,
            d_ff=8192,
            seq_length=1024,
            max_seq_len=1024,
            vocab_size=50272,
            micro_batch_size=2,
        )
        topology = DeviceTopology(
            device_specs=[
                DeviceConfig(device_id=0),
                DeviceConfig(device_id=1),
                DeviceConfig(device_id=2),
            ],
            bandwidths={
                (0, 1): 11440.0,
                (1, 0): 18640.0,
                (1, 2): 339.0,
                (2, 1): 1022.0,
                (0, 2): 341.0,
                (2, 0): 1062.0,
            },
            latencies={
                (0, 1): 0.1, (1, 0): 0.1,
                (1, 2): 0.1, (2, 1): 0.1,
                (0, 2): 0.1, (2, 0): 0.1,
            },
        )

        # Build a profiler with pairwise data
        profiler = _PairBWProfiler(
            per_rank_bw={
                0: 341.0 / 8000.0,
                1: 339.0 / 8000.0,
                2: 1022.0 / 8000.0,
            },
            pair_bw={
                (0, 1): 11440.0 / 8000.0,
                (1, 0): 18640.0 / 8000.0,
                (1, 2): 339.0 / 8000.0,
                (2, 1): 1022.0 / 8000.0,
                (0, 2): 341.0 / 8000.0,
                (2, 0): 1062.0 / 8000.0,
            },
        )

        s = ConfidentStrategy(pp_size=3, dp_size=1)
        # Stage 0→1 (fast link)
        fast_comm = s._comm_time(
            model_config, topology, 0, 0, 1, profiler,
        )
        # Stage 1→2 (slow link)
        slow_comm = s._comm_time(
            model_config, topology, 0, 1, 2, profiler,
        )

        # Fast link should be ~33x cheaper
        assert fast_comm < slow_comm / 5
        # Activation = 2 * 1024 * 2048 * 4 = 16 MB
        # Fast: 8 MB / 1.43 ≈ 5.6 ms
        assert fast_comm < 10.0
        # Slow: 8 MB / 0.042 ≈ 190 ms
        assert slow_comm > 100.0
