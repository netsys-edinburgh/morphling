"""Value types and calculations for sampled coordinator metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class NicCounters:
    bytes_sent: int
    bytes_recv: int


@dataclass(frozen=True, slots=True)
class SelectedNicCounters:
    nic_names: tuple[str, ...]
    counters: NicCounters


@dataclass(frozen=True, slots=True)
class MetricReading:
    timestamp_unix_s: float
    monotonic_s: float
    process_cpu_percent: float
    rss_bytes: int
    nic: NicCounters
    nic_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PhaseSnapshot:
    counts: Mapping[str, int]
    durations_seconds: Mapping[str, float]
    last_timestamps_unix_s: Mapping[str, float]

    @classmethod
    def empty(cls) -> PhaseSnapshot:
        return cls(counts={}, durations_seconds={}, last_timestamps_unix_s={})


@dataclass(frozen=True, slots=True)
class MetricsSample:
    reading: MetricReading
    nic_tx_bytes_per_sec: float
    nic_rx_bytes_per_sec: float
    phases: PhaseSnapshot

    def to_json_line(self) -> str:
        record = {
            "record_type": "sample",
            "timestamp_unix_s": self.reading.timestamp_unix_s,
            "process_cpu_percent": self.reading.process_cpu_percent,
            "rss_bytes": self.reading.rss_bytes,
            "nic_names": self.reading.nic_names,
            "nic_tx_bytes": self.reading.nic.bytes_sent,
            "nic_rx_bytes": self.reading.nic.bytes_recv,
            "nic_tx_bytes_per_sec": self.nic_tx_bytes_per_sec,
            "nic_rx_bytes_per_sec": self.nic_rx_bytes_per_sec,
            "phase_counts": self.phases.counts,
            "phase_durations_seconds": self.phases.durations_seconds,
            "phase_last_timestamps_unix_s": (
                self.phases.last_timestamps_unix_s
            ),
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"


def aggregate_nic_counters(
    counters: Mapping[str, NicCounters],
    nic_names: Sequence[str] | None,
) -> SelectedNicCounters:
    """Aggregate selected interfaces in stable lexical order."""
    selected_names = tuple(sorted(counters if nic_names is None else set(nic_names)))
    missing_names = tuple(name for name in selected_names if name not in counters)
    if missing_names:
        missing = ", ".join(missing_names)
        raise KeyError(f"network interface(s) unavailable: {missing}")
    return SelectedNicCounters(
        nic_names=selected_names,
        counters=NicCounters(
            bytes_sent=sum(counters[name].bytes_sent for name in selected_names),
            bytes_recv=sum(counters[name].bytes_recv for name in selected_names),
        ),
    )


def build_sample(
    current: MetricReading,
    previous: MetricReading | None,
    phases: PhaseSnapshot,
) -> MetricsSample:
    """Compute byte rates from cumulative NIC readings."""
    if previous is None:
        tx_rate = 0.0
        rx_rate = 0.0
    else:
        elapsed = current.monotonic_s - previous.monotonic_s
        if elapsed <= 0:
            tx_rate = 0.0
            rx_rate = 0.0
        else:
            tx_rate = (
                max(0, current.nic.bytes_sent - previous.nic.bytes_sent)
                / elapsed
            )
            rx_rate = (
                max(0, current.nic.bytes_recv - previous.nic.bytes_recv)
                / elapsed
            )
    return MetricsSample(
        reading=current,
        nic_tx_bytes_per_sec=tx_rate,
        nic_rx_bytes_per_sec=rx_rate,
        phases=phases,
    )
