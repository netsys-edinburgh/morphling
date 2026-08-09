"""Figure-builder and CLI tests for the coordinator-scaling pipeline.

These exercise the matplotlib rendering path with the ``Agg`` backend: the
three-panel figure structure (no figure-level title, concise panel labels,
canonical breakdown component order) and the CLI contract (clear failure on a
missing input path, successful dual PDF + PNG output from fixture JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.coord_scaling_plot import cli, figure, schema

# ── Fixture builders (mirror the real emitter field names) ──────────────────


def _summary() -> dict:
    def _row(device_count: int, seconds: float, cpu: float) -> dict:
        return {
            "device_count": device_count,
            "iteration_runtime_seconds": seconds,
            "median_cpu_percent": cpu,
            "peak_cpu_percent": 5500.0,
            "substrate": "single-host loopback/emulated devices",
        }

    return {
        "environment": {
            "logical_cpu_count": 56,
            "host_nic_substrate": "single-host loopback/emulated devices",
        },
        "rows": [
            _row(1, 12.0, 2200.0),
            _row(2, 9.6, 2600.0),
            _row(4, 9.7, 2950.0),
            _row(8, 9.9, 2860.0),
        ],
    }


def _scaling(mode: str, base: float, second: float) -> dict:
    def _point(coordinators: int, seconds: float) -> dict:
        return {
            "configuration": {"mode": mode, "coordinators": coordinators},
            "result": {
                "coordinators": coordinators,
                "iteration_total_seconds": seconds,
                "throughput_samples_per_second": 16.0 / seconds,
            },
        }

    return {
        "points": [_point(1, base), _point(2, second)],
        "conclusion": {"mode": mode, "efficiency": base / second},
    }


def _breakdown() -> dict:
    def _row(mode: str, coordinators: int, sync: float) -> dict:
        idle = 10.0 - 6.0 - sync - 1.0
        return {
            "mode": mode,
            "coordinators": coordinators,
            "iteration_total": 10.0,
            "device_dispatch_aggregation": 6.0,
            "gradient_sync": sync,
            "optimizer": 1.0,
            "idle_other": idle,
        }

    return {
        "component_semantics": {"gradient_sync": "AllReduce over loopback"},
        "rows": [
            _row("strong", 1, 0.5),
            _row("strong", 2, 1.5),
            _row("weak", 1, 0.5),
            _row("weak", 2, 1.5),
        ],
    }


def _parsed():
    device = schema.parse_device_scaling(_summary())
    strong = schema.parse_scaling_efficiency(
        _scaling("strong", 10.0, 6.0), "strong"
    )
    weak = schema.parse_scaling_efficiency(_scaling("weak", 8.0, 9.0), "weak")
    breakdown = schema.parse_breakdown(_breakdown())
    return device, strong, weak, breakdown


def _write_inputs(tmp_path: Path) -> dict:
    paths = {
        "summary": tmp_path / "summary.json",
        "strong": tmp_path / "strong.json",
        "weak": tmp_path / "weak.json",
        "breakdown": tmp_path / "breakdown.json",
    }
    paths["summary"].write_text(json.dumps(_summary()), encoding="utf-8")
    paths["strong"].write_text(
        json.dumps(_scaling("strong", 10.0, 6.0)), "utf-8"
    )
    paths["weak"].write_text(json.dumps(_scaling("weak", 8.0, 9.0)), "utf-8")
    paths["breakdown"].write_text(json.dumps(_breakdown()), encoding="utf-8")
    return paths


# ── Figure structure ────────────────────────────────────────────────────────


def test_build_figure_has_three_panels_and_no_figure_title() -> None:
    import matplotlib.pyplot as plt

    fig = figure.build_figure(*_parsed())
    try:
        titles = [ax.get_title() for ax in fig.axes]
        assert fig._suptitle is None
        assert any(title.startswith("(a)") for title in titles)
        assert any(title.startswith("(b)") for title in titles)
        assert any(title.startswith("(c)") for title in titles)
        # Figure is the 7.0-inch double-column width.
        assert fig.get_figwidth() == pytest.approx(7.0)
    finally:
        plt.close(fig)


def test_breakdown_panel_legend_uses_canonical_component_order() -> None:
    import matplotlib.pyplot as plt

    fig = figure.build_figure(*_parsed())
    try:
        assert len(figure.COMPONENT_LABELS) == len(schema.BREAKDOWN_COMPONENTS)
        panel_c = next(
            ax for ax in fig.axes if ax.get_title().startswith("(c)")
        )
        legend = panel_c.get_legend()
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == list(figure.COMPONENT_LABELS)
    finally:
        plt.close(fig)


# ── CLI contract ────────────────────────────────────────────────────────────


def test_cli_missing_input_path_fails_clearly(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "absent_summary.json"
    exit_code = cli.main(
        [
            "--summary",
            str(missing),
            "--strong",
            str(tmp_path / "strong.json"),
            "--weak",
            str(tmp_path / "weak.json"),
            "--breakdown",
            str(tmp_path / "breakdown.json"),
            "--out",
            str(tmp_path / "fig"),
        ]
    )

    assert exit_code == 2
    assert "absent_summary.json" in capsys.readouterr().err


def test_cli_generates_vector_pdf_and_png(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    out_base = tmp_path / "coordinator_scaling"

    exit_code = cli.main(
        [
            "--summary",
            str(inputs["summary"]),
            "--strong",
            str(inputs["strong"]),
            "--weak",
            str(inputs["weak"]),
            "--breakdown",
            str(inputs["breakdown"]),
            "--out",
            str(out_base),
        ]
    )

    assert exit_code == 0
    pdf = out_base.with_suffix(".pdf")
    png = out_base.with_suffix(".png")
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert png.is_file() and png.stat().st_size > 0
