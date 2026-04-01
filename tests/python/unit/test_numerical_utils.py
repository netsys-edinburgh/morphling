# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import pytest
import torch

from tests.python.testutils.numerical import (
    TIER_STRICT,
    assert_no_nan_inf,
    compare_loss_trajectories,
    compare_tensors,
    ulp_distance,
)


def test_compare_tensors_matching():
    actual = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    result = compare_tensors(actual, expected, TIER_STRICT)

    assert result.passed is True
    assert result.max_abs_diff == 0.0


def test_compare_tensors_divergent():
    actual = torch.tensor([1.0, 2.1, 3.0], dtype=torch.float32)
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    result = compare_tensors(actual, expected, TIER_STRICT)

    assert result.passed is False
    assert result.max_abs_diff > 0


def test_ulp_distance_known_values():
    a = torch.tensor([1.0], dtype=torch.float32)
    b = torch.nextafter(a, torch.tensor([2.0], dtype=torch.float32))

    distance = ulp_distance(a, b)

    assert distance.shape == a.shape
    assert int(distance.item()) == 1


def test_assert_no_nan_inf_clean():
    clean = torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32)

    assert_no_nan_inf(clean, name="clean_tensor")

    with pytest.raises(AssertionError, match="clean_tensor"):
        assert_no_nan_inf(
            torch.tensor([0.0, float("nan")], dtype=torch.float32),
            name="clean_tensor",
        )


def test_compare_loss_trajectories_matching():
    actual = [1.0, 0.5, 0.25]
    expected = [1.0, 0.5, 0.25]

    result = compare_loss_trajectories(actual, expected, TIER_STRICT)

    assert result.passed is True


def test_compare_loss_trajectories_divergent():
    actual = [1.0, 0.5, 0.1]
    expected = [1.0, 0.5, 0.25]

    result = compare_loss_trajectories(actual, expected, TIER_STRICT)

    assert result.passed is False
    assert result.num_mismatched > 0
