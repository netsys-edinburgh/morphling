"""Typed JSON boundary parsing for the coordinator-scaling figure.

Untrusted result JSON crosses into typed, immutable dataclasses exactly once,
here.  Everything downstream (the figure builder, the CLI) receives validated
values and never re-checks them.  This module deliberately depends only on the
standard library so it can be unit-tested without matplotlib.

Consumed inputs and the fields this parser actually reads:

* ``summary.json`` (single-coordinator device scaling, panel a) --
  ``environment.logical_cpu_count``, ``environment.host_nic_substrate`` and,
  per ``rows[]`` entry, ``device_count``, ``iteration_runtime_seconds`` and
  ``median_cpu_percent``.  Sustained CPU is normalized to a percentage of the
  full host by dividing process CPU percent by the logical CPU count.
* ``strong.json`` / ``weak.json`` (multi-coordinator scaling, panel b) --
  per ``points[]`` entry, ``configuration.coordinators`` and
  ``result.iteration_total_seconds``.  Efficiency is derived per point against
  the single-coordinator baseline: strong keeps global work fixed
  (``base / (coordinators * seconds)``); weak keeps local work fixed
  (``base / seconds``).
* ``breakdown.json`` (per-iteration composition, panel c) -- per ``rows[]``
  entry, ``mode``, ``coordinators``, ``iteration_total`` and the four
  components in :data:`BREAKDOWN_COMPONENTS`, which must reconcile to the
  iteration total.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

# Canonical bottom-to-top stack order for the per-iteration breakdown.
BREAKDOWN_COMPONENTS: Tuple[str, ...] = (
    "device_dispatch_aggregation",
    "gradient_sync",
    "optimizer",
    "idle_other",
)

SCALING_MODES: Tuple[str, ...] = ("strong", "weak")

_DEFAULT_SUBSTRATE = "single-host loopback/emulated devices"
_RECON_RTOL = 0.02
_RECON_ATOL = 1e-6


class SchemaError(ValueError):
    """Raised when result JSON is missing required fields or is malformed."""


class MissingDataError(SchemaError):
    """Raised when required JSON is well-formed but has no usable points."""


# ── Typed, immutable views ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DeviceScalingPoint:
    device_count: int
    iteration_runtime_seconds: float
    sustained_cpu_percent: float
    peak_cpu_percent: Optional[float]


@dataclass(frozen=True)
class DeviceScaling:
    substrate: str
    logical_cpu_count: int
    points: Tuple[DeviceScalingPoint, ...]


@dataclass(frozen=True)
class EfficiencyPoint:
    coordinators: int
    iteration_total_seconds: float
    efficiency: float
    throughput_samples_per_second: Optional[float]


@dataclass(frozen=True)
class ScalingEfficiency:
    mode: str
    points: Tuple[EfficiencyPoint, ...]


@dataclass(frozen=True)
class BreakdownRow:
    mode: str
    coordinators: int
    iteration_total: float
    components: Tuple[float, ...]


@dataclass(frozen=True)
class Breakdown:
    rows: Tuple[BreakdownRow, ...]
    component_semantics: Tuple[Tuple[str, str], ...]


# ── Small validation helpers ────────────────────────────────────────────────


def _mapping(value: Any, ctx: str) -> dict:
    if not isinstance(value, dict):
        raise SchemaError(
            f"{ctx}: expected an object, got {type(value).__name__}"
        )
    return value


def _sequence(value: Any, ctx: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{ctx}: expected a list, got {type(value).__name__}")
    return list(value)


def _require(mapping: dict, key: str, ctx: str) -> Any:
    if key not in mapping:
        raise SchemaError(f"{ctx}: missing required field '{key}'")
    return mapping[key]


def _as_float(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{ctx}: expected a number, got {value!r}")
    return float(value)


def _as_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{ctx}: expected an integer, got {value!r}")
    return value


def load_json(path: Any) -> Any:
    """Read and decode a JSON file, failing clearly on missing/invalid input."""
    resolved = Path(path)
    if not resolved.is_file():
        raise SchemaError(f"input file not found: {resolved}")
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON in {resolved}: {exc}") from exc


# ── Parsers ─────────────────────────────────────────────────────────────────


def parse_device_scaling(data: Any) -> DeviceScaling:
    """Parse single-coordinator device-scaling ``summary.json`` (panel a)."""
    root = _mapping(data, "summary")
    environment = _mapping(
        _require(root, "environment", "summary"), "summary.environment"
    )
    logical = _as_int(
        _require(environment, "logical_cpu_count", "summary.environment"),
        "logical_cpu_count",
    )
    if logical <= 0:
        raise SchemaError(
            "summary.environment.logical_cpu_count must be positive"
        )
    rows = _sequence(_require(root, "rows", "summary"), "summary.rows")

    points = []
    substrate = environment.get("host_nic_substrate")
    for index, entry in enumerate(rows):
        row = _mapping(entry, f"summary.rows[{index}]")
        ctx = f"summary.rows[{index}]"
        runtime = _require(row, "iteration_runtime_seconds", ctx)
        if runtime is None:  # failed repetition -- keep only measured points
            continue
        substrate = substrate or row.get("substrate")
        peak = row.get("peak_cpu_percent")
        points.append(
            DeviceScalingPoint(
                device_count=_as_int(_require(row, "device_count", ctx), ctx),
                iteration_runtime_seconds=_as_float(runtime, ctx),
                sustained_cpu_percent=_as_float(
                    _require(row, "median_cpu_percent", ctx), ctx
                )
                / logical,
                peak_cpu_percent=None if peak is None else _as_float(peak, ctx),
            )
        )

    if not points:
        raise MissingDataError("summary.rows has no measured device points")
    points.sort(key=lambda point: point.device_count)
    return DeviceScaling(
        substrate=substrate or _DEFAULT_SUBSTRATE,
        logical_cpu_count=logical,
        points=tuple(points),
    )


def parse_scaling_efficiency(data: Any, mode: str) -> ScalingEfficiency:
    """Parse ``strong.json``/``weak.json`` and derive per-point efficiency."""
    if mode not in SCALING_MODES:
        raise SchemaError(
            f"unknown scaling mode {mode!r}; expected {SCALING_MODES}"
        )
    root = _mapping(data, mode)
    entries = _sequence(_require(root, "points", mode), f"{mode}.points")

    seconds_by_coord = {}
    raw = []
    for index, entry in enumerate(entries):
        point = _mapping(entry, f"{mode}.points[{index}]")
        config = _mapping(
            _require(point, "configuration", f"{mode}.points[{index}]"),
            "configuration",
        )
        result = _mapping(
            _require(point, "result", f"{mode}.points[{index}]"), "result"
        )
        coordinators = _as_int(
            _require(config, "coordinators", "configuration"), "coordinators"
        )
        seconds = _as_float(
            _require(result, "iteration_total_seconds", "result"),
            "iteration_total_seconds",
        )
        throughput = result.get("throughput_samples_per_second")
        seconds_by_coord[coordinators] = seconds
        raw.append(
            (
                coordinators,
                seconds,
                None if throughput is None else _as_float(throughput, "result"),
            )
        )

    if not raw:
        raise MissingDataError(f"{mode}.points is empty")
    baseline = seconds_by_coord.get(1)
    if baseline is None:
        raise MissingDataError(
            f"{mode} scaling lacks the single-coordinator baseline point"
        )

    points = []
    for coordinators, seconds, throughput in sorted(
        raw, key=lambda item: item[0]
    ):
        if mode == "strong":
            efficiency = baseline / (coordinators * seconds)
        else:
            efficiency = baseline / seconds
        points.append(
            EfficiencyPoint(coordinators, seconds, efficiency, throughput)
        )
    return ScalingEfficiency(mode=mode, points=tuple(points))


def parse_breakdown(data: Any) -> Breakdown:
    """Parse ``breakdown.json`` stacked-bar components (panel c)."""
    root = _mapping(data, "breakdown")
    entries = _sequence(_require(root, "rows", "breakdown"), "breakdown.rows")
    if not entries:
        raise MissingDataError("breakdown.rows is empty")

    rows = []
    for index, entry in enumerate(entries):
        row = _mapping(entry, f"breakdown.rows[{index}]")
        ctx = f"breakdown.rows[{index}]"
        iteration_total = _as_float(_require(row, "iteration_total", ctx), ctx)
        components = tuple(
            _as_float(_require(row, name, ctx), ctx)
            for name in BREAKDOWN_COMPONENTS
        )
        if abs(sum(components) - iteration_total) > (
            _RECON_ATOL + _RECON_RTOL * abs(iteration_total)
        ):
            raise SchemaError(
                f"{ctx}: components {sum(components):.4f}s do not reconcile with "
                f"iteration_total {iteration_total:.4f}s"
            )
        rows.append(
            BreakdownRow(
                mode=str(_require(row, "mode", ctx)),
                coordinators=_as_int(_require(row, "coordinators", ctx), ctx),
                iteration_total=iteration_total,
                components=components,
            )
        )

    semantics = root.get("component_semantics", {})
    semantics_pairs = tuple(
        (str(key), str(value))
        for key, value in _mapping(
            semantics, "breakdown.component_semantics"
        ).items()
    )
    return Breakdown(rows=tuple(rows), component_semantics=semantics_pairs)
