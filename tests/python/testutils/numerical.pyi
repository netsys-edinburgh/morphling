from dataclasses import dataclass
from typing import List

import torch

@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    name: str
    description: str

TIER_STRICT: Tolerance
TIER_STANDARD: Tolerance
TIER_RELAXED: Tolerance
TIER_EXACT: Tolerance

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

def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor, tolerance: Tolerance
) -> ComparisonResult: ...
def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
def compare_loss_trajectories(
    actual: List[float], expected: List[float], tolerance: Tolerance
) -> ComparisonResult: ...
def assert_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None: ...
