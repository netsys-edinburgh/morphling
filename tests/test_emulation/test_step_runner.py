import threading
import time
from typing import Dict, List

import pytest

from morphling.emulation.barrier import StructuralBarrier
from morphling.emulation.dilation import DeviceDilation, DilationMatrix
from morphling.emulation.step_runner import (
    BSPStepRunner,
    EmulationConfig,
    EmulationStepResult,
)


def _sleep_fn(duration: float):
    def fn():
        time.sleep(duration)

    return fn


def _run_device(
    runner: BSPStepRunner,
    rank: int,
    compute_fn,
    comm_fn,
    results: Dict[int, EmulationStepResult],
):
    results[rank] = runner.run_step(0, compute_fn, comm_fn)


class TestSleepBasedDilation:
    def test_dilation_applied(self):
        dilation = DilationMatrix(
            devices=[
                DeviceDilation(rank=0, alpha_compute=2.0, beta_comm=1.0),
                DeviceDilation(rank=1, alpha_compute=1.0, beta_comm=1.0),
            ]
        )
        config = EmulationConfig(dilation=dilation, num_steps=1, num_devices=2)
        barrier = StructuralBarrier(2)

        results: Dict[int, EmulationStepResult] = {}
        threads = []
        for r in range(2):
            runner = BSPStepRunner(rank=r, config=config, barrier=barrier)
            t = threading.Thread(
                target=_run_device,
                args=(runner, r, _sleep_fn(0.01), _sleep_fn(0.005), results),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        r0 = results[0]
        r1 = results[1]

        ratio_0 = r0.virtual_compute_s[0] / r0.wall_compute_s[0]
        assert ratio_0 == pytest.approx(2.0, rel=0.01)

        ratio_1 = r1.virtual_compute_s[1] / r1.wall_compute_s[1]
        assert ratio_1 == pytest.approx(1.0, rel=0.01)

        assert r0.step_virtual_time_s > 0
        assert r1.step_virtual_time_s > 0


class TestOverlapModeFull:
    def test_overlap_reduces_total(self):
        dilation = DilationMatrix.uniform(1, alpha=1.0, beta=1.0)
        config = EmulationConfig(
            dilation=dilation, num_steps=1, num_devices=1, overlap_mode="full"
        )
        barrier = StructuralBarrier(1)
        runner = BSPStepRunner(rank=0, config=config, barrier=barrier)

        result = runner.run_step(0, _sleep_fn(0.02), _sleep_fn(0.01))

        vt_compute = result.virtual_compute_s[0]
        vt_comm = result.virtual_comm_s[0]
        vt_total = result.virtual_total_s[0]

        assert vt_total == pytest.approx(
            vt_compute + max(0.0, vt_comm - vt_compute), rel=0.01
        )
        assert vt_total < vt_compute + vt_comm


class TestToStepResult:
    def test_conversion(self):
        dilation = DilationMatrix.uniform(1)
        config = EmulationConfig(dilation=dilation, num_steps=1, num_devices=1)
        barrier = StructuralBarrier(1)
        runner = BSPStepRunner(rank=0, config=config, barrier=barrier)
        emu_result = runner.run_step(0, _sleep_fn(0.01), _sleep_fn(0.005))

        sr = emu_result.to_step_result()
        assert sr.fwd_time_s[0] == pytest.approx(
            emu_result.virtual_compute_s[0]
        )
        assert sr.bwd_time_s[0] == 0.0
        assert sr.grad_sync_time_s[0] == pytest.approx(
            emu_result.virtual_comm_s[0]
        )
        assert sr.total_time_s[0] == pytest.approx(
            emu_result.virtual_total_s[0]
        )
        assert sr.step_time_s == pytest.approx(emu_result.step_virtual_time_s)
