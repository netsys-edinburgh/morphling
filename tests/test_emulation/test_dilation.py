import pytest

from morphling.emulation.dilation import DeviceDilation, DilationMatrix


class TestDilateCompute:
    def test_applies_alpha(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0, alpha_compute=2.5)])
        assert dm.dilate_compute(0, 1.0) == 2.5

    def test_applies_alpha_fractional(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0, alpha_compute=0.4)])
        assert dm.dilate_compute(0, 2.0) == pytest.approx(0.8)


class TestDilateComm:
    def test_applies_beta(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0, beta_comm=0.3)])
        assert dm.dilate_comm(0, 1.0) == pytest.approx(0.3)


class TestUniformFactory:
    def test_creates_n_devices(self):
        dm = DilationMatrix.uniform(4, alpha=2.0, beta=0.5)
        assert len(dm.devices) == 4
        for d in dm.devices:
            assert d.alpha_compute == 2.0
            assert d.beta_comm == 0.5


class TestFromDictFactory:
    def test_roundtrip(self):
        dm = DilationMatrix.from_dict({0: (1.0, 1.0), 1: (3.0, 0.5)})
        assert dm.alpha(1) == 3.0
        assert dm.beta(1) == 0.5


class TestValidation:
    def test_rejects_zero_alpha(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0, alpha_compute=0.0)])
        with pytest.raises(ValueError, match="alpha_compute"):
            dm.validate()

    def test_rejects_negative_beta(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0, beta_comm=-1.0)])
        with pytest.raises(ValueError, match="beta_comm"):
            dm.validate()

    def test_rejects_duplicate_ranks(self):
        dm = DilationMatrix(
            devices=[
                DeviceDilation(rank=0),
                DeviceDilation(rank=0),
            ]
        )
        with pytest.raises(ValueError, match="Duplicate"):
            dm.validate()


class TestUnknownRank:
    def test_raises_key_error(self):
        dm = DilationMatrix(devices=[DeviceDilation(rank=0)])
        with pytest.raises(KeyError):
            dm.dilate_compute(99, 1.0)
