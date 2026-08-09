from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "morphling"
    / "runtime"
    / "coordinator_metrics.py"
)
CLI_MODULE_PATH = MODULE_PATH.with_name("coordinator_metrics_cli.py")
CLI_SPEC = importlib.util.spec_from_file_location(
    "morphling.runtime.coordinator_metrics_cli", CLI_MODULE_PATH
)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
coordinator_metrics_cli = importlib.util.module_from_spec(CLI_SPEC)
sys.modules[CLI_SPEC.name] = coordinator_metrics_cli
CLI_SPEC.loader.exec_module(coordinator_metrics_cli)
MODELS_MODULE_PATH = MODULE_PATH.with_name("coordinator_metrics_models.py")
MODELS_SPEC = importlib.util.spec_from_file_location(
    "morphling.runtime.coordinator_metrics_models", MODELS_MODULE_PATH
)
assert MODELS_SPEC is not None and MODELS_SPEC.loader is not None
coordinator_metrics_models = importlib.util.module_from_spec(MODELS_SPEC)
sys.modules[MODELS_SPEC.name] = coordinator_metrics_models
MODELS_SPEC.loader.exec_module(coordinator_metrics_models)
SPEC = importlib.util.spec_from_file_location(
    "coordinator_metrics", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
coordinator_metrics = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = coordinator_metrics
SPEC.loader.exec_module(coordinator_metrics)

MetricReading = coordinator_metrics.MetricReading
NicCounters = coordinator_metrics.NicCounters
PhaseRecorder = coordinator_metrics.PhaseRecorder
PhaseSnapshot = coordinator_metrics.PhaseSnapshot
MetricsConfig = coordinator_metrics.MetricsConfig
CoordinatorMetricsCollector = coordinator_metrics.CoordinatorMetricsCollector
aggregate_nic_counters = coordinator_metrics.aggregate_nic_counters
build_sample = coordinator_metrics.build_sample


class FakeSamplingError(RuntimeError):
    pass


class FakeMetricsReader:
    def __init__(
        self,
        readings: list[MetricReading | Exception],
        *,
        prime_error: Exception | None = None,
        read_attempted: threading.Event | None = None,
    ) -> None:
        self._readings = iter(readings)
        self._prime_error = prime_error
        self._read_attempted = read_attempted

    def prime(self) -> None:
        if self._prime_error is not None:
            raise self._prime_error

    def read(self, _nic_names: tuple[str, ...] | None) -> MetricReading:
        reading = next(self._readings)
        if isinstance(reading, Exception):
            if self._read_attempted is not None:
                self._read_attempted.set()
            raise reading
        return reading


def _reading(monotonic_s: float, sent: int, received: int) -> MetricReading:
    return MetricReading(
        timestamp_unix_s=100.0 + monotonic_s,
        monotonic_s=monotonic_s,
        process_cpu_percent=25.0,
        rss_bytes=1_000,
        nic=NicCounters(bytes_sent=sent, bytes_recv=received),
        nic_names=("eth0",),
    )


def test_metrics_cli_defaults_preserve_disabled_behavior() -> None:
    # Given
    parser = argparse.ArgumentParser()
    coordinator_metrics.add_metrics_arguments(parser)

    # When
    args = parser.parse_args([])

    # Then
    assert args.metrics_output is None
    assert args.metrics_interval == 1.0
    assert args.metrics_nics is None


def test_metrics_cli_parses_output_interval_and_nics(tmp_path: Path) -> None:
    # Given
    parser = argparse.ArgumentParser()
    coordinator_metrics.add_metrics_arguments(parser)
    output_path = tmp_path / "coordinator.jsonl"

    # When
    args = parser.parse_args(
        [
            "--metrics_output",
            str(output_path),
            "--metrics_interval",
            "0.5",
            "--metrics_nics",
            "ib0",
            "eth0",
        ]
    )

    # Then
    assert args.metrics_output == output_path
    assert args.metrics_interval == 0.5
    assert args.metrics_nics == ["ib0", "eth0"]


def test_multi_coordinator_output_paths_do_not_collide(tmp_path: Path) -> None:
    # Given
    output_directory = tmp_path / "metrics"

    # When
    rank_zero = coordinator_metrics.metrics_output_path(
        output_directory, rank=0
    )
    rank_one = coordinator_metrics.metrics_output_path(output_directory, rank=1)

    # Then
    assert rank_zero == output_directory / "coordinator-rank-0.jsonl"
    assert rank_one == output_directory / "coordinator-rank-1.jsonl"
    assert rank_zero != rank_one


def test_jsonl_sample_contains_resource_totals_and_computed_rates() -> None:
    # Given
    previous = MetricReading(
        timestamp_unix_s=100.0,
        monotonic_s=10.0,
        process_cpu_percent=25.0,
        rss_bytes=1_000,
        nic=NicCounters(bytes_sent=2_000, bytes_recv=4_000),
        nic_names=("eth0",),
    )
    current = MetricReading(
        timestamp_unix_s=102.0,
        monotonic_s=12.0,
        process_cpu_percent=50.0,
        rss_bytes=1_500,
        nic=NicCounters(bytes_sent=2_600, bytes_recv=5_000),
        nic_names=("eth0",),
    )

    # When
    line = build_sample(current, previous, PhaseSnapshot.empty()).to_json_line()
    sample = json.loads(line)

    # Then
    assert sample["timestamp_unix_s"] == 102.0
    assert sample["process_cpu_percent"] == 50.0
    assert sample["rss_bytes"] == 1_500
    assert sample["nic_tx_bytes"] == 2_600
    assert sample["nic_rx_bytes"] == 5_000
    assert sample["nic_tx_bytes_per_sec"] == 300.0
    assert sample["nic_rx_bytes_per_sec"] == 500.0


@pytest.mark.parametrize("current_monotonic_s", [10.0, 9.0])
def test_sample_uses_zero_rates_when_elapsed_is_nonpositive(
    current_monotonic_s: float,
) -> None:
    # Given
    previous = MetricReading(
        timestamp_unix_s=100.0,
        monotonic_s=10.0,
        process_cpu_percent=25.0,
        rss_bytes=1_000,
        nic=NicCounters(bytes_sent=2_000, bytes_recv=4_000),
        nic_names=("eth0",),
    )
    current = MetricReading(
        timestamp_unix_s=101.0,
        monotonic_s=current_monotonic_s,
        process_cpu_percent=50.0,
        rss_bytes=1_500,
        nic=NicCounters(bytes_sent=2_600, bytes_recv=5_000),
        nic_names=("eth0",),
    )

    # When
    sample = build_sample(current, previous, PhaseSnapshot.empty())

    # Then
    assert sample.nic_tx_bytes_per_sec == 0.0
    assert sample.nic_rx_bytes_per_sec == 0.0


def test_sample_clamps_rates_when_nic_counters_reset() -> None:
    # Given
    previous = _reading(monotonic_s=10.0, sent=2_000, received=4_000)
    current = _reading(monotonic_s=12.0, sent=100, received=200)

    # When
    sample = build_sample(current, previous, PhaseSnapshot.empty())

    # Then
    assert sample.nic_tx_bytes_per_sec == 0.0
    assert sample.nic_rx_bytes_per_sec == 0.0


def test_selected_nic_aggregation_is_deterministic() -> None:
    # Given
    counters = {
        "wlan0": NicCounters(bytes_sent=50, bytes_recv=60),
        "eth1": NicCounters(bytes_sent=20, bytes_recv=30),
        "eth0": NicCounters(bytes_sent=10, bytes_recv=40),
    }

    # When
    first = aggregate_nic_counters(counters, ("eth1", "eth0"))
    second = aggregate_nic_counters(counters, ("eth0", "eth1"))

    # Then
    assert first == second
    assert first.nic_names == ("eth0", "eth1")
    assert first.counters == NicCounters(bytes_sent=30, bytes_recv=70)


def test_phase_recorder_accumulates_without_per_event_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    recorder = PhaseRecorder()

    def fail_on_file_io(*_args: str, **_kwargs: str) -> None:
        raise AssertionError("phase recording must not perform file I/O")

    monkeypatch.setattr(Path, "open", fail_on_file_io)

    # When
    recorder.record("dispatch", duration_seconds=1.25, timestamp_unix_s=10.0)
    recorder.record("dispatch", duration_seconds=0.75, timestamp_unix_s=12.0)
    snapshot = recorder.snapshot()

    # Then
    assert snapshot.counts == {"dispatch": 2}
    assert snapshot.durations_seconds == {"dispatch": 2.0}
    assert snapshot.last_timestamps_unix_s == {"dispatch": 12.0}


def test_phase_track_records_when_body_raises() -> None:
    # Given
    recorder = PhaseRecorder()

    # When
    with pytest.raises(RuntimeError, match="training failed"):
        with recorder.track("iteration_total"):
            raise RuntimeError("training failed")

    # Then
    snapshot = recorder.snapshot()
    assert snapshot.counts == {"iteration_total": 1}
    assert snapshot.durations_seconds["iteration_total"] >= 0.0


def test_named_training_phases_accumulate_without_file_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    recorder = PhaseRecorder()
    phase_names = (
        "iteration_total",
        "forward_device_dispatch_aggregation",
        "backward_device_dispatch_aggregation",
        "gradient_sync",
        "optimizer",
        "idle_wait",
    )

    def fail_on_file_io(*_args: str, **_kwargs: str) -> None:
        raise AssertionError("phase tracking must not perform file I/O")

    monkeypatch.setattr(Path, "open", fail_on_file_io)

    # When
    for phase_name in phase_names:
        with coordinator_metrics.track_phase(recorder, phase_name):
            pass
    with coordinator_metrics.track_phase(recorder, "iteration_total"):
        pass
    snapshot = recorder.snapshot()

    # Then
    assert snapshot.counts == {
        "iteration_total": 2,
        "forward_device_dispatch_aggregation": 1,
        "backward_device_dispatch_aggregation": 1,
        "gradient_sync": 1,
        "optimizer": 1,
        "idle_wait": 1,
    }
    assert set(snapshot.durations_seconds) == set(phase_names)
    assert set(snapshot.last_timestamps_unix_s) == set(phase_names)


def test_collector_start_stop_writes_jsonl_and_stop_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    output_path = tmp_path / "metrics.jsonl"
    reader = FakeMetricsReader(
        [
            _reading(monotonic_s=10.0, sent=1_000, received=2_000),
            _reading(monotonic_s=12.0, sent=1_600, received=3_000),
        ]
    )
    monkeypatch.setattr(
        coordinator_metrics, "_PsutilMetricsReader", lambda: reader
    )
    collector = CoordinatorMetricsCollector(
        MetricsConfig(output_path=output_path, interval_seconds=60.0)
    )

    # When
    collector.start()
    collector.stop()
    collector.stop()
    samples = [
        json.loads(line) for line in output_path.read_text().splitlines()
    ]

    # Then
    assert [sample["record_type"] for sample in samples] == ["sample", "sample"]
    assert samples[1]["nic_tx_bytes_per_sec"] == 300.0
    assert samples[1]["nic_rx_bytes_per_sec"] == 500.0


def test_background_sampling_failure_writes_terminal_error_and_stop_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    output_path = tmp_path / "metrics.jsonl"
    read_attempted = threading.Event()
    reader = FakeMetricsReader(
        [
            _reading(monotonic_s=10.0, sent=1_000, received=2_000),
            FakeSamplingError("NIC sampling failed"),
        ],
        read_attempted=read_attempted,
    )
    monkeypatch.setattr(
        coordinator_metrics, "_PsutilMetricsReader", lambda: reader
    )
    collector = CoordinatorMetricsCollector(
        MetricsConfig(output_path=output_path, interval_seconds=0.001)
    )

    # When
    collector.start()
    assert read_attempted.wait(timeout=1.0)
    collector.stop()
    records = [
        json.loads(line) for line in output_path.read_text().splitlines()
    ]

    # Then
    assert set(records[-1]) == {
        "error_message",
        "error_type",
        "record_type",
        "timestamp_unix_s",
    }
    assert records[-1]["record_type"] == "terminal_error"
    assert records[-1]["error_type"] == "FakeSamplingError"
    assert records[-1]["error_message"] == "NIC sampling failed"
    assert collector.failure is not None


def test_start_closes_output_when_priming_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    output_path = tmp_path / "metrics.jsonl"
    reader = FakeMetricsReader(
        [], prime_error=FakeSamplingError("CPU priming failed")
    )
    monkeypatch.setattr(
        coordinator_metrics, "_PsutilMetricsReader", lambda: reader
    )
    collector = CoordinatorMetricsCollector(
        MetricsConfig(output_path=output_path, interval_seconds=1.0)
    )

    # When
    with pytest.raises(
        coordinator_metrics.MetricsCollectionError,
        match="CPU priming failed",
    ):
        collector.start()
    with output_path.open("a", encoding="utf-8") as output:
        output.write("")

    # Then
    assert collector._output is None
    terminal = json.loads(output_path.read_text().splitlines()[-1])
    assert terminal["record_type"] == "terminal_error"


def test_start_helper_reports_priming_failure_without_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Given
    reader = FakeMetricsReader(
        [], prime_error=FakeSamplingError("CPU priming failed")
    )
    monkeypatch.setattr(
        coordinator_metrics, "_PsutilMetricsReader", lambda: reader
    )
    config = MetricsConfig(
        output_path=tmp_path / "metrics.jsonl", interval_seconds=1.0
    )

    # When
    collector = coordinator_metrics.start_metrics_collector(
        config, PhaseRecorder()
    )

    # Then
    assert collector is None
    assert "CPU priming failed" in capsys.readouterr().err
