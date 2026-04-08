import pytest

from morphling.emulation.clock import DilatedDeviceClock
from morphling.emulation.dilation import DeviceDilation, DilationMatrix
from morphling.emulation.step_runner import EmulationStepResult


def _make_clock(alpha: float = 1.0, beta: float = 1.0) -> DilatedDeviceClock:
    dm = DilationMatrix(
        devices=[DeviceDilation(rank=0, alpha_compute=alpha, beta_comm=beta)]
    )
    return DilatedDeviceClock(rank=0, dilation=dm)


class TestClockFwdBwdSeparate:
    def test_records_both_phases(self):
        clock = _make_clock(alpha=2.0)
        clock.begin_step(0)
        fwd = clock.record_compute_fwd(0.010)
        bwd = clock.record_compute_bwd(0.020)
        assert fwd.phase == "compute_fwd"
        assert bwd.phase == "compute_bwd"
        assert fwd.virtual_time_s == pytest.approx(0.020)
        assert bwd.virtual_time_s == pytest.approx(0.040)
        assert fwd.dilation_factor == pytest.approx(2.0)
        assert bwd.dilation_factor == pytest.approx(2.0)
        assert clock.step_virtual_time_s == pytest.approx(0.060)


class TestStepResultFwdBwdMapping:
    def test_uses_split_when_available(self):
        result = EmulationStepResult(
            step_idx=0,
            virtual_compute_s={0: 0.050},
            virtual_compute_fwd_s={0: 0.020},
            virtual_compute_bwd_s={0: 0.030},
            virtual_comm_s={0: 0.010},
            virtual_total_s={0: 0.060},
            step_virtual_time_s=0.060,
        )
        sr = result.to_step_result()
        assert sr.fwd_time_s[0] == pytest.approx(0.020)
        assert sr.bwd_time_s[0] == pytest.approx(0.030)

    def test_falls_back_when_no_split(self):
        result = EmulationStepResult(
            step_idx=0,
            virtual_compute_s={0: 0.050},
            virtual_comm_s={0: 0.010},
            virtual_total_s={0: 0.060},
            step_virtual_time_s=0.060,
        )
        sr = result.to_step_result()
        assert sr.fwd_time_s[0] == pytest.approx(0.050)
        assert sr.bwd_time_s[0] == pytest.approx(0.0)


class TestFallbackNoGemmLog:
    def test_sleep_fn_no_split(self):
        import threading
        import time

        from morphling.emulation.barrier import StructuralBarrier
        from morphling.emulation.step_runner import (
            BSPStepRunner,
            EmulationConfig,
        )

        dilation = DilationMatrix.uniform(1)
        config = EmulationConfig(dilation=dilation, num_steps=1, num_devices=1)
        barrier = StructuralBarrier(1)
        runner = BSPStepRunner(rank=0, config=config, barrier=barrier)
        result = runner.run_step(
            0, lambda: time.sleep(0.005), lambda: time.sleep(0.002)
        )
        assert result.virtual_compute_s[0] > 0
        assert len(result.virtual_compute_fwd_s) == 0
        assert len(result.virtual_compute_bwd_s) == 0
