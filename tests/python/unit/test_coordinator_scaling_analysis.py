from __future__ import annotations

import json
from pathlib import Path

from scripts import coordinator_scaling_analysis


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_parse_metrics_uses_measured_iteration_window(tmp_path: Path) -> None:
    # Given
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "record_type": "sample",
                "process_cpu_percent": 10.0,
                "rss_bytes": 100,
                "nic_tx_bytes_per_sec": 10.0,
                "nic_rx_bytes_per_sec": 20.0,
                "phase_counts": {},
                "phase_durations_seconds": {},
            },
            {
                "record_type": "sample",
                "process_cpu_percent": 120.0,
                "rss_bytes": 300,
                "nic_tx_bytes_per_sec": 300.0,
                "nic_rx_bytes_per_sec": 500.0,
                "phase_counts": {"iteration_total": 1, "optimizer": 1},
                "phase_durations_seconds": {
                    "iteration_total": 3.0,
                    "optimizer": 0.25,
                },
            },
            {
                "record_type": "sample",
                "process_cpu_percent": 80.0,
                "rss_bytes": 250,
                "nic_tx_bytes_per_sec": 500.0,
                "nic_rx_bytes_per_sec": 700.0,
                "phase_counts": {"iteration_total": 2, "optimizer": 2},
                "phase_durations_seconds": {
                    "iteration_total": 8.0,
                    "optimizer": 0.5,
                },
            },
        ],
    )

    # When
    metrics = coordinator_scaling_analysis.parse_metrics_jsonl(metrics_path)

    # Then
    assert metrics.iteration_runtime_seconds == 4.0
    assert metrics.peak_cpu_percent == 120.0
    assert metrics.median_cpu_percent == 100.0
    assert metrics.peak_rss_bytes == 300
    assert metrics.peak_nic_tx_bytes_per_sec == 500.0
    assert metrics.median_nic_tx_bytes_per_sec == 400.0
    assert metrics.peak_nic_rx_bytes_per_sec == 700.0
    assert metrics.median_nic_rx_bytes_per_sec == 600.0
    assert metrics.phase_counts == {"iteration_total": 2, "optimizer": 2}
    assert metrics.phase_durations_seconds == {
        "iteration_total": 8.0,
        "optimizer": 0.5,
    }


def test_parse_metrics_retains_terminal_error(tmp_path: Path) -> None:
    # Given
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "record_type": "terminal_error",
                "error_type": "OSError",
                "error_message": "sampling failed",
                "timestamp_unix_s": 1.0,
            }
        ],
    )

    # When
    metrics = coordinator_scaling_analysis.parse_metrics_jsonl(metrics_path)

    # Then
    assert metrics.error == "OSError: sampling failed"
    assert metrics.iteration_runtime_seconds is None


def test_parse_metrics_skips_blank_lines(tmp_path: Path) -> None:
    # Given
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n  \n"
        + json.dumps(
            {
                "record_type": "sample",
                "process_cpu_percent": 10.0,
                "rss_bytes": 100,
                "nic_tx_bytes_per_sec": 1.0,
                "nic_rx_bytes_per_sec": 2.0,
                "phase_counts": {"iteration_total": 1},
                "phase_durations_seconds": {"iteration_total": 3.0},
            }
        )
        + "\n\n",
        encoding="utf-8",
    )

    # When
    metrics = coordinator_scaling_analysis.parse_metrics_jsonl(metrics_path)

    # Then
    assert metrics.error is None
    assert metrics.iteration_runtime_seconds == 3.0


def test_parse_metrics_warns_for_trailing_partial_record(tmp_path: Path) -> None:
    # Given
    metrics_path = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics_path,
        [
            {
                "record_type": "sample",
                "process_cpu_percent": 10.0,
                "rss_bytes": 100,
                "nic_tx_bytes_per_sec": 1.0,
                "nic_rx_bytes_per_sec": 2.0,
                "phase_counts": {"iteration_total": 1},
                "phase_durations_seconds": {"iteration_total": 3.0},
            }
        ],
    )
    with metrics_path.open("a", encoding="utf-8") as output:
        output.write('{"record_type":')

    # When
    metrics = coordinator_scaling_analysis.parse_metrics_jsonl(metrics_path)

    # Then
    assert metrics.error is None
    assert metrics.iteration_runtime_seconds == 3.0
    assert len(metrics.warnings) == 1
    assert "trailing partial" in metrics.warnings[0]


def test_parse_metrics_rejects_malformed_non_trailing_record(
    tmp_path: Path,
) -> None:
    # Given
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        '{"record_type":\n'
        '{"record_type":"sample","process_cpu_percent":10.0}\n',
        encoding="utf-8",
    )

    # When
    metrics = coordinator_scaling_analysis.parse_metrics_jsonl(metrics_path)

    # Then
    assert metrics.iteration_runtime_seconds is None
    assert metrics.error is not None
    assert "malformed metrics JSONL" in metrics.error


def test_aggregate_repetitions_reports_means_and_success_count() -> None:
    # Given
    first = coordinator_scaling_analysis.RunMeasurement(
        iteration_runtime_seconds=4.0,
        peak_cpu_percent=120.0,
        median_cpu_percent=100.0,
        peak_rss_bytes=300,
        peak_nic_tx_bytes_per_sec=500.0,
        median_nic_tx_bytes_per_sec=400.0,
        peak_nic_rx_bytes_per_sec=700.0,
        median_nic_rx_bytes_per_sec=600.0,
        phase_counts={"iteration_total": 2},
        phase_durations_seconds={"iteration_total": 8.0},
        error=None,
    )
    second = coordinator_scaling_analysis.RunMeasurement(
        iteration_runtime_seconds=6.0,
        peak_cpu_percent=140.0,
        median_cpu_percent=120.0,
        peak_rss_bytes=500,
        peak_nic_tx_bytes_per_sec=700.0,
        median_nic_tx_bytes_per_sec=600.0,
        peak_nic_rx_bytes_per_sec=900.0,
        median_nic_rx_bytes_per_sec=800.0,
        phase_counts={"iteration_total": 2},
        phase_durations_seconds={"iteration_total": 12.0},
        error=None,
    )

    # When
    aggregate = coordinator_scaling_analysis.aggregate_repetitions(
        device_count=4,
        measurements=(first, second),
        exit_statuses=(0, 0),
    )

    # Then
    assert aggregate.device_count == 4
    assert aggregate.repetitions == 2
    assert aggregate.successful_repetitions == 2
    assert aggregate.iteration_runtime_seconds == 5.0
    assert aggregate.peak_cpu_percent == 140.0
    assert aggregate.median_cpu_percent == 110.0
    assert aggregate.phase_counts == {"iteration_total": 2.0}
    assert aggregate.phase_durations_seconds == {"iteration_total": 10.0}


def test_runtime_plateau_without_sustained_cpu_does_not_name_cpu_bottleneck() -> None:
    # Given
    rows = (
        coordinator_scaling_analysis.ScalingPoint(1, 8.0, 100.0, 200.0),
        coordinator_scaling_analysis.ScalingPoint(2, 5.0, 150.0, 300.0),
        coordinator_scaling_analysis.ScalingPoint(4, 4.8, 200.0, 760.0),
        coordinator_scaling_analysis.ScalingPoint(8, 4.7, 220.0, 780.0),
    )

    # When
    assessment = coordinator_scaling_analysis.assess_saturation(
        rows, logical_cpu_count=8
    )

    # Then
    assert assessment.runtime_plateau_observed is True
    assert assessment.resource_saturation_observed is False
    assert assessment.bottleneck == (
        "coordinator-side host work / specific resource unisolated"
    )
    assert assessment.median_cpu_normalized_percent == 27.5
    assert assessment.peak_cpu_normalized_percent == 97.5


def test_sustained_cpu_and_runtime_plateau_are_reported_separately() -> None:
    # Given
    rows = (
        coordinator_scaling_analysis.ScalingPoint(1, 8.0, 100.0, 200.0),
        coordinator_scaling_analysis.ScalingPoint(2, 5.0, 200.0, 300.0),
        coordinator_scaling_analysis.ScalingPoint(4, 4.8, 720.0, 760.0),
        coordinator_scaling_analysis.ScalingPoint(8, 4.7, 736.0, 780.0),
    )

    # When
    assessment = coordinator_scaling_analysis.assess_saturation(
        rows, logical_cpu_count=8
    )

    # Then
    assert assessment.runtime_plateau_observed is True
    assert assessment.resource_saturation_observed is True
    assert assessment.bottleneck == "coordinator CPU"


def test_scaling_without_runtime_plateau_reports_only_measured_range() -> None:
    # Given
    rows = (
        coordinator_scaling_analysis.ScalingPoint(1, 8.0, 100.0, 200.0),
        coordinator_scaling_analysis.ScalingPoint(2, 5.0, 150.0, 300.0),
        coordinator_scaling_analysis.ScalingPoint(4, 3.0, 200.0, 400.0),
        coordinator_scaling_analysis.ScalingPoint(8, 2.0, 250.0, 500.0),
    )

    # When
    assessment = coordinator_scaling_analysis.assess_saturation(
        rows, logical_cpu_count=8
    )

    # Then
    assert assessment.runtime_plateau_observed is False
    assert assessment.bottleneck == (
        "none within measured range through 8 local emulated devices"
    )


def test_response_counter_validation_accepts_partitioned_decrements() -> None:
    # Given
    records = (
        (0, 1, 3, 2),
        (0, 1, 2, 1),
        (0, 1, 1, 0),
        (1, 1, 2, 1),
        (1, 1, 1, 0),
    )

    # When
    valid = coordinator_scaling_analysis.response_counters_are_balanced(records)

    # Then
    assert valid is True
