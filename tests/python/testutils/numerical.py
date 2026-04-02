from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    name: str
    description: str


TIER_STRICT = Tolerance(
    rtol=1e-5, atol=1e-5, name="strict", description="logits/activations"
)
TIER_STANDARD = Tolerance(
    rtol=1e-2, atol=1e-4, name="standard", description="loss trajectory"
)
TIER_RELAXED = Tolerance(
    rtol=1e-1, atol=1e-2, name="relaxed", description="gradient norms"
)
TIER_EXACT = Tolerance(
    rtol=0.0, atol=0.0, name="exact", description="determinism"
)


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    ulp_p50: float
    ulp_p95: float
    ulp_p99: float
    num_mismatched: int
    total_elements: int


def _as_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.float32:
        return tensor.to(torch.float32)
    return tensor


def _percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.to(torch.float32), q).item())


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute element-wise ULP distance between two float32 tensors."""
    if a.shape != b.shape:
        raise ValueError("ULP distance requires tensors with the same shape")

    a32 = _as_float_tensor(a).contiguous()
    b32 = _as_float_tensor(b).contiguous()

    a_bits = a32.view(torch.int32).to(torch.int64)
    b_bits = b32.view(torch.int32).to(torch.int64)

    sign_bit = 0x80000000
    a_ordered = torch.where(a_bits < 0, sign_bit - a_bits, a_bits + sign_bit)
    b_ordered = torch.where(b_bits < 0, sign_bit - b_bits, b_bits + sign_bit)

    return torch.abs(a_ordered - b_ordered)


def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor, tolerance: Tolerance
) -> ComparisonResult:
    """Compare two tensors, return ComparisonResult. Does NOT raise on failure."""
    actual_f = _as_float_tensor(actual)
    expected_f = _as_float_tensor(expected)

    passed = True
    try:
        torch.testing.assert_close(
            actual_f, expected_f, rtol=tolerance.rtol, atol=tolerance.atol
        )
    except AssertionError:
        passed = False

    if actual_f.shape != expected_f.shape:
        total_elements = max(actual_f.numel(), expected_f.numel())
        return ComparisonResult(
            passed=passed,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            ulp_p50=float("inf"),
            ulp_p95=float("inf"),
            ulp_p99=float("inf"),
            num_mismatched=total_elements,
            total_elements=total_elements,
        )

    diff = actual_f - expected_f
    abs_diff = torch.abs(diff)
    denom = torch.clamp(torch.abs(expected_f), min=1e-12)
    rel_diff = abs_diff / denom
    ulp = ulp_distance(actual_f, expected_f).to(torch.float32)

    mismatched = int(
        (
            ~torch.isclose(
                actual_f, expected_f, rtol=tolerance.rtol, atol=tolerance.atol
            )
        )
        .sum()
        .item()
    )
    total_elements = int(actual_f.numel())

    return ComparisonResult(
        passed=passed,
        max_abs_diff=float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        max_rel_diff=float(rel_diff.max().item()) if rel_diff.numel() else 0.0,
        ulp_p50=_percentile(ulp, 0.50),
        ulp_p95=_percentile(ulp, 0.95),
        ulp_p99=_percentile(ulp, 0.99),
        num_mismatched=mismatched,
        total_elements=total_elements,
    )


def compare_loss_trajectories(
    actual: list[float], expected: list[float], tolerance: Tolerance
) -> ComparisonResult:
    """Compare per-step loss lists. Returns ComparisonResult over the full trajectory."""
    return compare_tensors(
        torch.tensor(actual, dtype=torch.float32),
        torch.tensor(expected, dtype=torch.float32),
        tolerance,
    )


def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Raise AssertionError with descriptive message if tensor contains NaN or Inf."""
    if torch.isnan(tensor).any():
        raise AssertionError(f"Tensor '{name}' contains NaN")
    if torch.isinf(tensor).any():
        raise AssertionError(f"Tensor '{name}' contains Inf")
