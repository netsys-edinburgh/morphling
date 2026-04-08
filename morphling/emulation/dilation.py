"""Per-device, per-phase dilation factors for hybrid emulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class DeviceDilation:
    """Dilation factors for a single device."""

    rank: int
    alpha_compute: float = (
        1.0  # compute time multiplier (>1 = slower, <1 = faster)
    )
    beta_comm: float = 1.0  # communication time multiplier


@dataclass
class DilationMatrix:
    """Per-device x per-phase dilation factors for all devices."""

    devices: List[DeviceDilation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_rank: Dict[int, DeviceDilation] = {
            d.rank: d for d in self.devices
        }

    def alpha(self, rank: int) -> float:
        """Return compute dilation factor for *rank*."""
        return self._by_rank[rank].alpha_compute

    def beta(self, rank: int) -> float:
        """Return communication dilation factor for *rank*."""
        return self._by_rank[rank].beta_comm

    def dilate_compute(self, rank: int, wall_time_s: float) -> float:
        """Apply compute dilation: ``wall_time_s * alpha(rank)``."""
        return wall_time_s * self.alpha(rank)

    def dilate_comm(self, rank: int, wall_time_s: float) -> float:
        """Apply communication dilation: ``wall_time_s * beta(rank)``."""
        return wall_time_s * self.beta(rank)

    @classmethod
    def uniform(
        cls, num_devices: int, alpha: float = 1.0, beta: float = 1.0
    ) -> DilationMatrix:
        """Create uniform dilation (same factors for all devices)."""
        devices = [
            DeviceDilation(rank=r, alpha_compute=alpha, beta_comm=beta)
            for r in range(num_devices)
        ]
        return cls(devices=devices)

    @classmethod
    def from_dict(
        cls, factors: Dict[int, Tuple[float, float]]
    ) -> DilationMatrix:
        """Create from ``{rank: (alpha, beta)}`` mapping."""
        devices = [
            DeviceDilation(rank=r, alpha_compute=a, beta_comm=b)
            for r, (a, b) in factors.items()
        ]
        return cls(devices=devices)

    def validate(self) -> None:
        """Raise :class:`ValueError` if any factor is <= 0."""
        for d in self.devices:
            if d.alpha_compute <= 0.0:
                raise ValueError(
                    f"Device {d.rank}: alpha_compute must be > 0, got {d.alpha_compute}"
                )
            if d.beta_comm <= 0.0:
                raise ValueError(
                    f"Device {d.rank}: beta_comm must be > 0, got {d.beta_comm}"
                )
        ranks = [d.rank for d in self.devices]
        if len(ranks) != len(set(ranks)):
            raise ValueError(f"Duplicate ranks in dilation matrix: {ranks}")
