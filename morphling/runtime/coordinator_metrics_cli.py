"""CLI and output-path configuration for coordinator metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


class MetricsConfigurationError(ValueError):
    """Report invalid coordinator metrics configuration."""


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    output_path: Path
    interval_seconds: float = 1.0
    nic_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise MetricsConfigurationError(
                "metrics interval must be greater than zero"
            )


class MetricsCliArgs(Protocol):
    metrics_output: Path | None
    metrics_interval: float
    metrics_nics: Sequence[str] | None


def _positive_interval(raw: str) -> float:
    try:
        interval = float(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "metrics interval must be a number"
        ) from error
    if interval <= 0:
        raise argparse.ArgumentTypeError(
            "metrics interval must be greater than zero"
        )
    return interval


def add_metrics_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared opt-in coordinator metrics flags."""
    parser.add_argument(
        "--metrics_output",
        type=Path,
        default=None,
        help=(
            "Metrics destination: a path ending in .jsonl is a file; "
            "every other path is treated as a directory."
        ),
    )
    parser.add_argument(
        "--metrics_interval",
        type=_positive_interval,
        default=1.0,
        help="Sampling interval in seconds; must be greater than zero.",
    )
    parser.add_argument(
        "--metrics_nics",
        nargs="+",
        default=None,
        help="Optional network interface names to aggregate.",
    )


def metrics_output_path(destination: Path, rank: int | None = None) -> Path:
    """Resolve .jsonl files directly and all other paths as directories."""
    if rank is None:
        if destination.suffix == ".jsonl":
            return destination
        return destination / "coordinator.jsonl"
    if destination.suffix == ".jsonl":
        return destination.with_name(
            f"{destination.stem}-rank-{rank}{destination.suffix}"
        )
    return destination / f"coordinator-rank-{rank}.jsonl"


def metrics_config_from_args(
    args: MetricsCliArgs, rank: int | None = None
) -> MetricsConfig | None:
    """Build validated metrics configuration when collection is enabled."""
    if args.metrics_output is None:
        return None
    return MetricsConfig(
        output_path=metrics_output_path(args.metrics_output, rank=rank),
        interval_seconds=args.metrics_interval,
        nic_names=(
            tuple(args.metrics_nics) if args.metrics_nics is not None else None
        ),
    )
