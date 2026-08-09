from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "morphling"
    / "runtime"
    / "coordinator_metrics_cli.py"
)
SPEC = importlib.util.spec_from_file_location(
    "coordinator_metrics_cli", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
coordinator_metrics_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coordinator_metrics_cli
SPEC.loader.exec_module(coordinator_metrics_cli)


@pytest.mark.parametrize("interval", ["0", "-0.5"])
def test_cli_rejects_nonpositive_interval(interval: str) -> None:
    # Given
    parser = argparse.ArgumentParser()
    coordinator_metrics_cli.add_metrics_arguments(parser)

    # When / Then
    with pytest.raises(SystemExit):
        parser.parse_args(["--metrics_interval", interval])


@pytest.mark.parametrize("interval", [0.0, -0.5])
def test_config_rejects_nonpositive_interval(
    tmp_path: Path, interval: float
) -> None:
    # Given / When / Then
    with pytest.raises(
        coordinator_metrics_cli.MetricsConfigurationError,
        match="greater than zero",
    ):
        coordinator_metrics_cli.MetricsConfig(
            output_path=tmp_path / "metrics.jsonl",
            interval_seconds=interval,
        )


def test_output_path_preserves_explicit_jsonl_file(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "metrics.jsonl"

    # When
    output_path = coordinator_metrics_cli.metrics_output_path(destination)

    # Then
    assert output_path == destination


def test_output_path_uses_default_file_for_directory(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "metrics"

    # When
    output_path = coordinator_metrics_cli.metrics_output_path(destination)

    # Then
    assert output_path == destination / "coordinator.jsonl"


def test_ranked_output_path_suffixes_explicit_jsonl_file(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "metrics.jsonl"

    # When
    output_path = coordinator_metrics_cli.metrics_output_path(destination, rank=2)

    # Then
    assert output_path == tmp_path / "metrics-rank-2.jsonl"


def test_ranked_output_path_uses_rank_file_for_directory(tmp_path: Path) -> None:
    # Given
    destination = tmp_path / "metrics.results"

    # When
    output_path = coordinator_metrics_cli.metrics_output_path(destination, rank=2)

    # Then
    assert output_path == destination / "coordinator-rank-2.jsonl"


def test_config_from_args_keeps_metrics_disabled() -> None:
    # Given
    args = argparse.Namespace(
        metrics_output=None,
        metrics_interval=1.0,
        metrics_nics=None,
    )

    # When
    config = coordinator_metrics_cli.metrics_config_from_args(args)

    # Then
    assert config is None


def test_config_from_args_resolves_ranked_output_and_nics(tmp_path: Path) -> None:
    # Given
    args = argparse.Namespace(
        metrics_output=tmp_path / "metrics.jsonl",
        metrics_interval=0.5,
        metrics_nics=["ib0", "eth0"],
    )

    # When
    config = coordinator_metrics_cli.metrics_config_from_args(args, rank=3)

    # Then
    assert config == coordinator_metrics_cli.MetricsConfig(
        output_path=tmp_path / "metrics-rank-3.jsonl",
        interval_seconds=0.5,
        nic_names=("ib0", "eth0"),
    )
