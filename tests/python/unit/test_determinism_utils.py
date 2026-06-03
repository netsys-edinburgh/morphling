import pytest
import torch

from tests.python.testutils.determinism import (
    deterministic_context,
    set_all_seeds,
)

pytestmark = pytest.mark.smoke


def _cuda_same_seed_same_output() -> None:
    set_all_seeds(42)
    first = torch.randn(10, 10, device="cuda")
    set_all_seeds(42)
    second = torch.randn(10, 10, device="cuda")
    assert torch.equal(first, second)


def test_same_seed_same_output():
    set_all_seeds(42)
    first = torch.randn(10, 10)
    set_all_seeds(42)
    second = torch.randn(10, 10)
    assert torch.equal(first, second)

    if torch.cuda.is_available():
        _cuda_same_seed_same_output()


def test_different_seed_different_output():
    set_all_seeds(42)
    first = torch.randn(10, 10)
    set_all_seeds(43)
    second = torch.randn(10, 10)
    assert not torch.equal(first, second)


def test_deterministic_context_sets_flags():
    before_deterministic = torch.backends.cudnn.deterministic
    before_benchmark = torch.backends.cudnn.benchmark

    with deterministic_context():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    assert torch.backends.cudnn.deterministic == before_deterministic
    assert torch.backends.cudnn.benchmark == before_benchmark
