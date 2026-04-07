import pytest

from morphling.emulation.clock import DilatedDeviceClock
from morphling.emulation.dilation import DeviceDilation, DilationMatrix


def _make_clock(alpha: float = 1.0, beta: float = 1.0) -> DilatedDeviceClock:
    dm = DilationMatrix(
        devices=[DeviceDilation(rank=0, alpha_compute=alpha, beta_comm=beta)]
    )
    return DilatedDeviceClock(rank=0, dilation=dm)


class TestRecordCompute:
    def test_returns_dilated_time(self):
        clock = _make_clock(alpha=3.0)
        clock.begin_step(0)
        rec = clock.record_compute(0.010)
        assert rec.phase == "compute"
        assert rec.wall_time_s == pytest.approx(0.010)
        assert rec.virtual_time_s == pytest.approx(0.030)
        assert rec.dilation_factor == pytest.approx(3.0)


class TestRecordComm:
    def test_returns_dilated_time(self):
        clock = _make_clock(beta=0.5)
        clock.begin_step(0)
        rec = clock.record_comm(0.020)
        assert rec.virtual_time_s == pytest.approx(0.010)


class TestStepAccumulation:
    def test_accumulates_phases(self):
        clock = _make_clock(alpha=2.0, beta=1.0)
        clock.begin_step(0)
        clock.record_compute(0.010)
        clock.record_comm(0.005)
        assert clock.step_virtual_time_s == pytest.approx(0.025)

    def test_begin_step_resets(self):
        clock = _make_clock(alpha=2.0)
        clock.begin_step(0)
        clock.record_compute(0.010)
        assert clock.step_virtual_time_s == pytest.approx(0.020)
        clock.begin_step(1)
        assert clock.step_virtual_time_s == pytest.approx(0.0)


class TestTotalVirtualTime:
    def test_spans_steps(self):
        clock = _make_clock(alpha=1.0, beta=1.0)
        clock.begin_step(0)
        clock.record_compute(0.030)
        clock.begin_step(1)
        clock.record_compute(0.020)
        assert clock.total_virtual_time_s() == pytest.approx(0.050)


class TestRecordWithoutBeginStep:
    def test_raises_runtime_error(self):
        clock = _make_clock()
        with pytest.raises(RuntimeError, match="begin_step"):
            clock.record_compute(0.01)
