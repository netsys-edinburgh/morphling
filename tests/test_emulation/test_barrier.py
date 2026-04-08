import threading

import pytest

from morphling.emulation.barrier import BarrierResult, StructuralBarrier

NUM_DEVICES = 4


def _run_barrier(
    barrier: StructuralBarrier,
    rank: int,
    phase: str,
    step_idx: int,
    vt: float,
    results: dict,
):
    results[rank] = barrier.barrier(rank, phase, step_idx, vt)


class TestBarrier4Threads:
    def test_all_get_same_result(self):
        barrier = StructuralBarrier(NUM_DEVICES)
        results: dict[int, BarrierResult] = {}
        threads = []
        for i in range(NUM_DEVICES):
            t = threading.Thread(
                target=_run_barrier,
                args=(barrier, i, "compute", 0, i * 0.01, results),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        assert len(results) == NUM_DEVICES
        expected_vt = {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.03}
        for rank, br in results.items():
            assert br.virtual_times_s == pytest.approx(expected_vt)
            assert br.max_virtual_time_s == pytest.approx(0.03)
            assert br.phase == "compute"
            assert br.step_idx == 0


class TestBarrierReusableAcrossPhases:
    def test_two_phases(self):
        barrier = StructuralBarrier(2)
        for phase in ("compute", "comm"):
            results: dict[int, BarrierResult] = {}
            threads = []
            for i in range(2):
                t = threading.Thread(
                    target=_run_barrier,
                    args=(barrier, i, phase, 0, float(i), results),
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            assert len(results) == 2
            assert results[0].phase == phase


class TestBarrierWallTime:
    def test_positive(self):
        barrier = StructuralBarrier(2)
        results: dict[int, BarrierResult] = {}
        threads = []
        for i in range(2):
            t = threading.Thread(
                target=_run_barrier,
                args=(barrier, i, "compute", 0, 0.0, results),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        for br in results.values():
            assert br.wall_barrier_time_s > 0


class TestBarrierReusableAcrossSteps:
    def test_two_steps(self):
        barrier = StructuralBarrier(2)
        for step_idx in range(2):
            results: dict[int, BarrierResult] = {}
            threads = []
            for i in range(2):
                t = threading.Thread(
                    target=_run_barrier,
                    args=(barrier, i, "compute", step_idx, 0.0, results),
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            assert results[0].step_idx == step_idx
