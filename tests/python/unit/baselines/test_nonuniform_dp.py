"""Tests for non-uniform DP runtime support."""
from __future__ import annotations

import pytest

from baselines.core.config import ParallelismPlan
from baselines.train import (
    _build_micro_routing,
    _build_rank_maps,
    _is_nonuniform_dp,
)


# ── hpp_plan.json fixture ────────────────────────────────

@pytest.fixture
def nonuniform_plan() -> ParallelismPlan:
    """Plan matching current hpp_plan.json (5 stages, 6 ranks)."""
    return ParallelismPlan(
        partition_points=[1, 2, 12, 13],
        device_groups={
            0: [0], 1: [1], 2: [2], 3: [3, 4], 4: [5],
        },
        micro_batch_alloc={
            0: {0: 2},
            1: {1: 2},
            2: {2: 2},
            3: {3: 1, 4: 1},
            4: {5: 2},
        },
        schedule_type="1f1b",
        estimated_latency_ms=7209.78,
    )


@pytest.fixture
def uniform_plan() -> ParallelismPlan:
    """Uniform 3-stage, 2-DP plan (6 ranks)."""
    return ParallelismPlan(
        partition_points=[8, 16],
        device_groups={
            0: [0, 3], 1: [1, 4], 2: [2, 5],
        },
        micro_batch_alloc={
            0: {0: 2, 3: 2},
            1: {1: 2, 4: 2},
            2: {2: 2, 5: 2},
        },
        schedule_type="1f1b",
    )


# ── _is_nonuniform_dp ───────────────────────────────────

class TestIsNonuniform:
    def test_nonuniform(self, nonuniform_plan):
        assert _is_nonuniform_dp(nonuniform_plan) is True

    def test_uniform(self, uniform_plan):
        assert _is_nonuniform_dp(uniform_plan) is False

    def test_empty(self):
        plan = ParallelismPlan()
        assert _is_nonuniform_dp(plan) is False


# ── _build_rank_maps ────────────────────────────────────

class TestBuildRankMaps:
    def test_nonuniform(self, nonuniform_plan):
        r2s, r2d, sdp = _build_rank_maps(nonuniform_plan)
        # rank 0 → stage 0, dp_idx 0
        assert r2s[0] == 0
        assert r2d[0] == 0
        # rank 3 → stage 3, dp_idx 0
        assert r2s[3] == 3
        assert r2d[3] == 0
        # rank 4 → stage 3, dp_idx 1
        assert r2s[4] == 3
        assert r2d[4] == 1
        # rank 5 → stage 4, dp_idx 0
        assert r2s[5] == 4
        assert r2d[5] == 0
        # stage dp sizes
        assert sdp[0] == 1
        assert sdp[3] == 2
        assert sdp[4] == 1

    def test_uniform(self, uniform_plan):
        r2s, r2d, sdp = _build_rank_maps(uniform_plan)
        # rank 0 → stage 0, dp_idx 0
        assert r2s[0] == 0
        assert r2d[0] == 0
        # rank 3 → stage 0, dp_idx 1
        assert r2s[3] == 0
        assert r2d[3] == 1
        # All stages have dp_size 2
        for s in sdp:
            assert sdp[s] == 2


# ── _build_micro_routing ────────────────────────────────

class TestMicroRouting:
    def test_rank0_stage0(self, nonuniform_plan):
        """Rank 0 (stage 0): sends to rank 1 (stage 1)."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            nonuniform_plan, rank=0, my_stage=0,
            num_stages=5, num_micro=2,
        )
        assert my_m == [0, 1]
        # Stage 0 is first → no recv
        assert fwd_recv == {}
        # Stage 0 sends to stage 1 (rank 1)
        assert fwd_send[0] == 1
        assert fwd_send[1] == 1

    def test_rank3_stage3_dp0(self, nonuniform_plan):
        """Rank 3 (stage 3, dp_idx 0): gets micro 0 only."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            nonuniform_plan, rank=3, my_stage=3,
            num_stages=5, num_micro=2,
        )
        assert my_m == [0]
        # Recv micro 0 from stage 2 (rank 2)
        assert fwd_recv[0] == 2
        # Send micro 0 to stage 4 (rank 5)
        assert fwd_send[0] == 5

    def test_rank4_stage3_dp1(self, nonuniform_plan):
        """Rank 4 (stage 3, dp_idx 1): gets micro 1 only."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            nonuniform_plan, rank=4, my_stage=3,
            num_stages=5, num_micro=2,
        )
        assert my_m == [1]
        # Recv micro 1 from stage 2 (rank 2)
        assert fwd_recv[1] == 2
        # Send micro 1 to stage 4 (rank 5)
        assert fwd_send[1] == 5

    def test_rank2_stage2(self, nonuniform_plan):
        """Rank 2 (stage 2): sends micro 0→rank3, micro 1→rank4."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            nonuniform_plan, rank=2, my_stage=2,
            num_stages=5, num_micro=2,
        )
        assert my_m == [0, 1]
        # Recv from stage 1 (rank 1)
        assert fwd_recv[0] == 1
        assert fwd_recv[1] == 1
        # Send: micro 0 → rank 3, micro 1 → rank 4
        assert fwd_send[0] == 3
        assert fwd_send[1] == 4

    def test_rank5_stage4_last(self, nonuniform_plan):
        """Rank 5 (stage 4, last): receives from rank 3 and 4."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            nonuniform_plan, rank=5, my_stage=4,
            num_stages=5, num_micro=2,
        )
        assert my_m == [0, 1]
        # Recv micro 0 from rank 3, micro 1 from rank 4
        assert fwd_recv[0] == 3
        assert fwd_recv[1] == 4
        # Last stage → no send
        assert fwd_send == {}


# ── Weight-based micro allocation (production scale) ────

@pytest.fixture
def four_stage_plan() -> ParallelismPlan:
    """4-stage plan matching current cluster deployment.

    Stage 0: [0,1] (DP=2), Stage 1: [2], Stage 2: [3,4] (DP=2), Stage 3: [5]
    micro_batch_alloc values are WEIGHTS (1:1 = equal split).
    """
    return ParallelismPlan(
        partition_points=[1, 11, 12],
        device_groups={
            0: [0, 1], 1: [2], 2: [3, 4], 3: [5],
        },
        micro_batch_alloc={
            0: {0: 1, 1: 1},
            1: {2: 2},
            2: {3: 1, 4: 1},
            3: {5: 2},
        },
        schedule_type="1f1b",
        estimated_latency_ms=7574.42,
    )


class TestWeightBasedAlloc:
    def test_equal_split_16_micros(self, four_stage_plan):
        """Stage 0 with {0:1, 1:1} and 16 micros → 8 each."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            four_stage_plan, rank=0, my_stage=0,
            num_stages=4, num_micro=16,
        )
        assert my_m == list(range(0, 8))
        assert fwd_recv == {}  # first stage
        # All go to stage 1 (rank 2)
        for mid in my_m:
            assert fwd_send[mid] == 2

    def test_rank1_gets_second_half(self, four_stage_plan):
        """Rank 1 (stage 0, dp_idx 1): gets micros 8-15."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            four_stage_plan, rank=1, my_stage=0,
            num_stages=4, num_micro=16,
        )
        assert my_m == list(range(8, 16))
        for mid in my_m:
            assert fwd_send[mid] == 2

    def test_rank2_all_16_micros(self, four_stage_plan):
        """Rank 2 (stage 1, solo): gets all 16 micros."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            four_stage_plan, rank=2, my_stage=1,
            num_stages=4, num_micro=16,
        )
        assert my_m == list(range(16))
        # Recv micros 0-7 from rank 0, 8-15 from rank 1
        for mid in range(0, 8):
            assert fwd_recv[mid] == 0
        for mid in range(8, 16):
            assert fwd_recv[mid] == 1
        # Send micros 0-7 to rank 3, 8-15 to rank 4
        for mid in range(0, 8):
            assert fwd_send[mid] == 3
        for mid in range(8, 16):
            assert fwd_send[mid] == 4

    def test_rank3_stage2_first_half(self, four_stage_plan):
        """Rank 3 (stage 2, dp_idx 0): gets micros 0-7."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            four_stage_plan, rank=3, my_stage=2,
            num_stages=4, num_micro=16,
        )
        assert my_m == list(range(0, 8))
        for mid in my_m:
            assert fwd_recv[mid] == 2  # from rank 2
            assert fwd_send[mid] == 5  # to rank 5

    def test_rank5_receives_from_both(self, four_stage_plan):
        """Rank 5 (stage 3, last): receives 0-7 from rank 3, 8-15 from rank 4."""
        my_m, fwd_recv, fwd_send = _build_micro_routing(
            four_stage_plan, rank=5, my_stage=3,
            num_stages=4, num_micro=16,
        )
        assert my_m == list(range(16))
        for mid in range(0, 8):
            assert fwd_recv[mid] == 3
        for mid in range(8, 16):
            assert fwd_recv[mid] == 4
        assert fwd_send == {}  # last stage

    def test_unequal_weights(self):
        """Weights 1:3 on 16 micros → 4 and 12."""
        plan = ParallelismPlan(
            partition_points=[12],
            device_groups={0: [0, 1], 1: [2]},
            micro_batch_alloc={
                0: {0: 1, 1: 3},
                1: {2: 4},
            },
            schedule_type="1f1b",
        )
        my_m0, _, _ = _build_micro_routing(
            plan, rank=0, my_stage=0,
            num_stages=2, num_micro=16,
        )
        my_m1, _, _ = _build_micro_routing(
            plan, rank=1, my_stage=0,
            num_stages=2, num_micro=16,
        )
        assert len(my_m0) == 4
        assert len(my_m1) == 12
        assert my_m0 == list(range(0, 4))
        assert my_m1 == list(range(4, 16))
