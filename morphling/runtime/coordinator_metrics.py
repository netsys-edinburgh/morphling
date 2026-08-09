"""Low-overhead sampled metrics for coordinator training processes."""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Protocol, TextIO

import psutil

from morphling.runtime.coordinator_metrics_cli import (
    MetricsConfig,
    add_metrics_arguments,
    metrics_config_from_args,
    metrics_output_path,
)
from morphling.runtime.coordinator_metrics_models import (
    MetricReading,
    MetricsSample,
    NicCounters,
    PhaseSnapshot,
    SelectedNicCounters,
    aggregate_nic_counters,
    build_sample,
)


class PhaseRecorder:
    """Accumulate phase events in memory for sampled publication."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._durations: dict[str, float] = {}
        self._timestamps: dict[str, float] = {}

    def record(
        self, name: str, *, duration_seconds: float, timestamp_unix_s: float
    ) -> None:
        with self._lock:
            self._counts[name] = self._counts.get(name, 0) + 1
            self._durations[name] = (
                self._durations.get(name, 0.0) + duration_seconds
            )
            self._timestamps[name] = timestamp_unix_s

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record(
                name,
                duration_seconds=time.perf_counter() - started,
                timestamp_unix_s=time.time(),
            )

    def snapshot(self) -> PhaseSnapshot:
        with self._lock:
            return PhaseSnapshot(
                counts=dict(self._counts),
                durations_seconds=dict(self._durations),
                last_timestamps_unix_s=dict(self._timestamps),
            )


@contextmanager
def track_phase(
    recorder: PhaseRecorder | None, name: str
) -> Iterator[None]:
    """Track a phase only when metrics collection is enabled."""
    if recorder is None:
        yield
    else:
        with recorder.track(name):
            yield


class MetricsCollectionError(RuntimeError):
    """Represent a terminal collector failure without stopping training."""

    def __init__(self, cause: Exception) -> None:
        self.error_type = type(cause).__name__
        super().__init__(str(cause))


class _MetricsReader(Protocol):
    def prime(self) -> None: ...

    def read(self, nic_names: tuple[str, ...] | None) -> MetricReading: ...


class _PsutilMetricsReader:
    def __init__(self) -> None:
        self._process = psutil.Process()

    def prime(self) -> None:
        self._process.cpu_percent(interval=None)

    def read(self, nic_names: tuple[str, ...] | None) -> MetricReading:
        per_nic = psutil.net_io_counters(pernic=True)
        selected = aggregate_nic_counters(
            {
                name: NicCounters(value.bytes_sent, value.bytes_recv)
                for name, value in per_nic.items()
            },
            nic_names,
        )
        with self._process.oneshot():
            return MetricReading(
                timestamp_unix_s=time.time(),
                monotonic_s=time.monotonic(),
                process_cpu_percent=self._process.cpu_percent(interval=None),
                rss_bytes=self._process.memory_info().rss,
                nic=selected.counters,
                nic_names=selected.nic_names,
            )


class CoordinatorMetricsCollector:
    """Sample psutil metrics on a background thread and append JSONL."""

    def __init__(
        self, config: MetricsConfig, phase_recorder: PhaseRecorder | None = None
    ) -> None:
        self._config = config
        self._phases = phase_recorder or PhaseRecorder()
        self._reader: _MetricsReader = _PsutilMetricsReader()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output: TextIO | None = None
        self._previous: MetricReading | None = None
        self._failure: MetricsCollectionError | None = None

    @property
    def failure(self) -> MetricsCollectionError | None:
        return self._failure

    def start(self) -> None:
        self._config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output = self._config.output_path.open("w", encoding="utf-8")
        try:
            self._reader.prime()
            self._sample_once()
        except Exception as error:  # noqa: BLE001 - thread/process boundary
            failure = self._record_failure(error)
            self._close_output()
            raise failure from error
        self._thread = threading.Thread(
            target=self._run,
            name="coordinator-metrics",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._output is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._failure is None:
            try:
                self._sample_once()
            except Exception as error:  # noqa: BLE001 - shutdown boundary
                self._record_failure(error)
        self._close_output()
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._config.interval_seconds):
            try:
                self._sample_once()
            except Exception as error:  # noqa: BLE001 - daemon boundary
                self._record_failure(error)
                self._stop_event.set()
                return

    def _sample_once(self) -> None:
        reading = self._reader.read(self._config.nic_names)
        sample = build_sample(reading, self._previous, self._phases.snapshot())
        self._previous = reading
        if self._output is not None:
            self._output.write(sample.to_json_line())
            self._output.flush()

    def _record_failure(self, error: Exception) -> MetricsCollectionError:
        if self._failure is not None:
            return self._failure
        failure = MetricsCollectionError(error)
        self._failure = failure
        terminal = {
            "record_type": "terminal_error",
            "timestamp_unix_s": time.time(),
            "error_type": failure.error_type,
            "error_message": str(failure),
        }
        if self._output is not None:
            try:
                self._output.write(
                    json.dumps(terminal, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                self._output.flush()
            except (OSError, ValueError) as output_error:
                print(
                    f"Coordinator metrics terminal write failed: {output_error}",
                    file=sys.stderr,
                    flush=True,
                )
        print(f"Coordinator metrics stopped: {failure}", file=sys.stderr, flush=True)
        return failure

    def _close_output(self) -> None:
        output = self._output
        self._output = None
        if output is None:
            return
        try:
            output.flush()
        except (OSError, ValueError) as output_error:
            print(
                f"Coordinator metrics flush failed: {output_error}",
                file=sys.stderr,
                flush=True,
            )
        try:
            output.close()
        except (OSError, ValueError) as output_error:
            print(
                f"Coordinator metrics close failed: {output_error}",
                file=sys.stderr,
                flush=True,
            )


def start_metrics_collector(
    config: MetricsConfig, phase_recorder: PhaseRecorder
) -> CoordinatorMetricsCollector | None:
    """Start metrics without allowing setup failure to abort training."""
    collector = CoordinatorMetricsCollector(config, phase_recorder)
    try:
        collector.start()
    except MetricsCollectionError:
        return None
    return collector
