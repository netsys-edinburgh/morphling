"""Unit tests for the coordinator-scaling plot data-prep boundary parser.

These tests exercise ``scripts.coord_scaling_plot.schema`` in isolation
(no matplotlib): typed dataclass construction, JSON boundary validation,
per-point scaling-efficiency computation, and stacked-breakdown component
ordering / reconciliation.  Every fixture is a temp JSON built in-memory so
the tests never touch real result files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.coord_scaling_plot import schema

# ── Fixture builders ────────────────────────────────────────────────────────


def _summary(logical_cpu_count: int = 50) -> dict:
    return {
        "schema_version": 1,
        "environment": {
            "logical_cpu_count": logical_cpu_count,
            "host_nic_substrate": "single-host loopback/emulated devices",
        },
        "configuration": {"device_counts": [1, 2]},
        "rows": [
            {
                "device_count": 2,
                "iteration_runtime_seconds": 9.6,
                "median_cpu_percent": 2600.0,
                "peak_cpu_percent": 5500.0,
                "substrate": "single-host loopback/emulated devices",
            },
            {
                "device_count": 1,
                "iteration_runtime_seconds": 12.0,
                "median_cpu_percent": 2200.0,
                "peak_cpu_percent": 5560.0,
                "substrate": "single-host loopback/emulated devices",
            },
        ],
    }


def _scaling(mode: str, base_seconds: float, second_seconds: float) -> dict:
    def _point(coordinators: int, total_devices: int, seconds: float) -> dict:
        return {
            "configuration": {
                "mode": mode,
                "coordinators": coordinators,
                "total_devices": total_devices,
            },
            "result": {
                "coordinators": coordinators,
                "iteration_total_seconds": seconds,
                "throughput_samples_per_second": 16.0 / seconds,
            },
        }

    return {
        "schema_version": 1,
        "experiment": f"{mode} multi-coordinator scaling",
        "points": [
            _point(1, 8 if mode == "strong" else 4, base_seconds),
            _point(2, 8, second_seconds),
        ],
        "conclusion": {"mode": mode, "efficiency": 0.5, "speedup": 1.0},
    }


def _breakdown() -> dict:
    def _row(mode: str, coordinators: int) -> dict:
        # Deliberately scrambled key order to prove canonical reordering.
        return {
            "idle_other": 1.5,
            "coordinators": coordinators,
            "optimizer": 1.0,
            "mode": mode,
            "gradient_sync": 1.5,
            "iteration_total": 10.0,
            "device_dispatch_aggregation": 6.0,
        }

    return {
        "schema_version": 1,
        "component_semantics": {"gradient_sync": "AllReduce over loopback"},
        "rows": [
            _row("strong", 1),
            _row("strong", 2),
            _row("weak", 1),
            _row("weak", 2),
        ],
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── Device-scaling parsing (panel a) ────────────────────────────────────────


def test_parse_device_scaling_sorts_points_and_normalizes_cpu() -> None:
    # Given a summary with rows out of device-count order
    device = schema.parse_device_scaling(_summary(logical_cpu_count=50))

    # Then points are sorted ascending and CPU is normalized by logical CPUs
    assert [p.device_count for p in device.points] == [1, 2]
    assert device.points[0].sustained_cpu_percent == pytest.approx(44.0)
    assert device.points[1].sustained_cpu_percent == pytest.approx(52.0)
    assert device.points[0].iteration_runtime_seconds == pytest.approx(12.0)
    assert "loopback" in device.substrate
    assert device.logical_cpu_count == 50


def test_parse_device_scaling_missing_runtime_field_raises_schema_error() -> (
    None
):
    payload = _summary()
    del payload["rows"][0]["iteration_runtime_seconds"]

    with pytest.raises(schema.SchemaError):
        schema.parse_device_scaling(payload)


def test_parse_device_scaling_missing_logical_cpu_raises_schema_error() -> None:
    payload = _summary()
    del payload["environment"]["logical_cpu_count"]

    with pytest.raises(schema.SchemaError):
        schema.parse_device_scaling(payload)


def test_parse_device_scaling_empty_rows_raises_missing_data() -> None:
    payload = _summary()
    payload["rows"] = []

    with pytest.raises(schema.MissingDataError):
        schema.parse_device_scaling(payload)


def test_parse_device_scaling_skips_null_runtime_points() -> None:
    payload = _summary()
    payload["rows"][0]["iteration_runtime_seconds"] = None

    device = schema.parse_device_scaling(payload)

    assert [p.device_count for p in device.points] == [1]


# ── Scaling efficiency parsing (panel b) ────────────────────────────────────


def test_strong_efficiency_computed_per_point_with_unit_baseline() -> None:
    # Given strong points: base=10s @1 coord, 6s @2 coords
    strong = schema.parse_scaling_efficiency(
        _scaling("strong", base_seconds=10.0, second_seconds=6.0), "strong"
    )

    # Then efficiency(1)=1.0 and efficiency(2)=base/(2*meas)=10/12
    assert strong.mode == "strong"
    assert [p.coordinators for p in strong.points] == [1, 2]
    assert strong.points[0].efficiency == pytest.approx(1.0)
    assert strong.points[1].efficiency == pytest.approx(10.0 / 12.0)


def test_weak_efficiency_uses_local_baseline() -> None:
    weak = schema.parse_scaling_efficiency(
        _scaling("weak", base_seconds=8.0, second_seconds=9.0), "weak"
    )

    assert weak.points[0].efficiency == pytest.approx(1.0)
    assert weak.points[1].efficiency == pytest.approx(8.0 / 9.0)


def test_scaling_efficiency_missing_baseline_raises_missing_data() -> None:
    payload = _scaling("strong", 10.0, 6.0)
    payload["points"] = [payload["points"][1]]  # only coordinators=2

    with pytest.raises(schema.MissingDataError):
        schema.parse_scaling_efficiency(payload, "strong")


def test_scaling_efficiency_rejects_unknown_mode() -> None:
    with pytest.raises(schema.SchemaError):
        schema.parse_scaling_efficiency(_scaling("strong", 10.0, 6.0), "bogus")


# ── Breakdown parsing (panel c) ─────────────────────────────────────────────


def test_breakdown_orders_components_canonically() -> None:
    breakdown = schema.parse_breakdown(_breakdown())

    assert schema.BREAKDOWN_COMPONENTS == (
        "device_dispatch_aggregation",
        "gradient_sync",
        "optimizer",
        "idle_other",
    )
    row = breakdown.rows[0]
    # components tuple is aligned to BREAKDOWN_COMPONENTS regardless of key order
    assert row.components == pytest.approx((6.0, 1.5, 1.0, 1.5))
    assert [(r.mode, r.coordinators) for r in breakdown.rows] == [
        ("strong", 1),
        ("strong", 2),
        ("weak", 1),
        ("weak", 2),
    ]


def test_breakdown_components_reconcile_to_iteration_total() -> None:
    breakdown = schema.parse_breakdown(_breakdown())

    row = breakdown.rows[0]
    assert sum(row.components) == pytest.approx(row.iteration_total)


def test_breakdown_non_reconciling_row_raises_schema_error() -> None:
    payload = _breakdown()
    payload["rows"][0]["idle_other"] = 99.0  # sum no longer == iteration_total

    with pytest.raises(schema.SchemaError):
        schema.parse_breakdown(payload)


def test_breakdown_missing_component_raises_schema_error() -> None:
    payload = _breakdown()
    del payload["rows"][0]["gradient_sync"]

    with pytest.raises(schema.SchemaError):
        schema.parse_breakdown(payload)


def test_breakdown_empty_rows_raises_missing_data() -> None:
    payload = _breakdown()
    payload["rows"] = []

    with pytest.raises(schema.MissingDataError):
        schema.parse_breakdown(payload)


# ── JSON loading boundary ───────────────────────────────────────────────────


def test_load_json_missing_file_raises_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"

    with pytest.raises(schema.SchemaError) as excinfo:
        schema.load_json(missing)

    assert "nope.json" in str(excinfo.value)


def test_load_json_invalid_json_raises_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(schema.SchemaError):
        schema.load_json(bad)


def test_load_json_roundtrip(tmp_path: Path) -> None:
    path = _write(tmp_path, "summary.json", _summary())

    assert schema.load_json(path)["schema_version"] == 1
