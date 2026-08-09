"""Three-panel double-column coordinator-scaling figure (paper figure D8).

Panels, left to right:

* (a) Measured iteration runtime and sustained normalized CPU versus emulated
  device count, with the loopback/emulated substrate and the measured device
  range marked and no exact value labels.
* (b) Strong and weak multi-coordinator scaling efficiency for coordinator
  counts 1 and 2, as separate marked lines against an ideal reference; only
  the available points are drawn.
* (c) Per-iteration time composition as stacked bars for each available
  (mode, coordinator-count) run, using the canonical component order that
  reconciles to the iteration total.

Rendering uses the conference-plot skill's paper style, Wong colorblind-safe
palette and hatches (via :mod:`._plot_utils`).  The figure carries no
figure-level title; each panel gets a concise ``(a)/(b)/(c)`` label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import schema
from ._plot_utils import HATCHES, WONG_PALETTE, paper_style, save_paper_figure

COMPONENT_LABELS: Tuple[str, ...] = (
    "Dispatch+agg", "Gradient sync", "Optimizer", "Idle/other")

_FIG_WIDTH_INCHES = 7.0
_FIG_HEIGHT_INCHES = 2.4
_SMALL = 6

_RUNTIME_COLOR = WONG_PALETTE[5]
_CPU_COLOR = WONG_PALETTE[6]
_STRONG_COLOR = WONG_PALETTE[1]
_WEAK_COLOR = WONG_PALETTE[3]
_COMPONENT_COLORS = (
    WONG_PALETTE[2], WONG_PALETTE[3], WONG_PALETTE[6], WONG_PALETTE[7])
_COMPONENT_HATCHES = (HATCHES[1], HATCHES[2], HATCHES[3], HATCHES[4])


def _panel_device(ax, device: schema.DeviceScaling) -> None:
    counts = [point.device_count for point in device.points]
    runtimes = [point.iteration_runtime_seconds for point in device.points]
    cpu = [point.sustained_cpu_percent for point in device.points]

    line_rt = ax.plot(counts, runtimes, color=_RUNTIME_COLOR, marker="o",
                      markersize=3.5, label="Iteration runtime")
    twin = ax.twinx()
    line_cpu = twin.plot(counts, cpu, color=_CPU_COLOR, marker="s", ls="--",
                         markersize=3.5, label="Sustained CPU")

    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(count) for count in counts])
    ax.set_xlim(min(counts) * 2 ** -0.4, max(counts) * 2 ** 0.4)
    ax.axvspan(min(counts), max(counts), color="0.85", alpha=0.35, zorder=0)
    ax.set_xlabel("Emulated devices")
    ax.set_ylabel("Iteration runtime (s)", color=_RUNTIME_COLOR)
    ax.tick_params(axis="y", colors=_RUNTIME_COLOR)
    ax.set_ylim(bottom=0)
    twin.set_ylabel("Sustained CPU (% host)", color=_CPU_COLOR)
    twin.tick_params(axis="y", colors=_CPU_COLOR)
    twin.set_ylim(bottom=0)
    twin.grid(False)

    ax.text(0.5, 0.12, "loopback/emulated substrate", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=_SMALL, style="italic", color="0.35")
    ax.text(0.5, 0.03, "measured range", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=_SMALL, color="0.4")
    handles = line_rt + line_cpu
    ax.legend(handles, [handle.get_label() for handle in handles],
              fontsize=_SMALL, loc="center right", framealpha=0.9)


def _plot_efficiency(ax, efficiency: schema.ScalingEfficiency, color: str,
                     marker: str, style: str) -> None:
    coordinators = [point.coordinators for point in efficiency.points]
    values = [point.efficiency for point in efficiency.points]
    ax.plot(coordinators, values, color=color, marker=marker, ls=style,
            markersize=4, label=efficiency.mode.capitalize())


def _panel_efficiency(ax, strong: schema.ScalingEfficiency,
                      weak: schema.ScalingEfficiency) -> None:
    _plot_efficiency(ax, strong, _STRONG_COLOR, "o", "-")
    _plot_efficiency(ax, weak, _WEAK_COLOR, "s", "--")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, label="Ideal")

    ticks = sorted({point.coordinators for point in (*strong.points, *weak.points)})
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks])
    ax.set_xlim(min(ticks) - 0.2, max(ticks) + 0.2)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("Coordinators")
    ax.set_ylabel("Scaling efficiency")
    ax.legend(fontsize=_SMALL, loc="lower left", framealpha=0.9)


def _panel_breakdown(ax, breakdown: schema.Breakdown) -> None:
    rows = breakdown.rows
    positions = np.arange(len(rows))
    labels = [("S" if row.mode == "strong" else "W") + str(row.coordinators)
              for row in rows]
    bottoms = np.zeros(len(rows))
    for index, component_label in enumerate(COMPONENT_LABELS):
        heights = np.array([row.components[index] for row in rows])
        ax.bar(positions, heights, bottom=bottoms, width=0.7,
               color=_COMPONENT_COLORS[index], hatch=_COMPONENT_HATCHES[index],
               edgecolor="black", linewidth=0.4, label=component_label)
        bottoms += heights

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mode-coordinators")
    ax.set_ylabel("Per-iteration time (s)")
    top = max((row.iteration_total for row in rows), default=1.0)
    ax.set_ylim(0, top * 1.35)
    ax.legend(fontsize=_SMALL, loc="upper right", framealpha=0.9,
              handlelength=1.4, handleheight=1.2)


def _assemble(device: schema.DeviceScaling, strong: schema.ScalingEfficiency,
              weak: schema.ScalingEfficiency,
              breakdown: schema.Breakdown):
    fig, axes = plt.subplots(1, 3)
    _panel_device(axes[0], device)
    _panel_efficiency(axes[1], strong, weak)
    _panel_breakdown(axes[2], breakdown)
    axes[0].set_title("(a) Coordinator device scaling")
    axes[1].set_title("(b) Multi-coordinator efficiency")
    axes[2].set_title("(c) Per-iteration breakdown")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig


def build_figure(device: schema.DeviceScaling, strong: schema.ScalingEfficiency,
                 weak: schema.ScalingEfficiency,
                 breakdown: schema.Breakdown):
    """Build and return the three-panel figure (caller owns ``plt.close``)."""
    with paper_style(width=_FIG_WIDTH_INCHES, height=_FIG_HEIGHT_INCHES,
                     font_size=7):
        return _assemble(device, strong, weak, breakdown)


def render(device: schema.DeviceScaling, strong: schema.ScalingEfficiency,
           weak: schema.ScalingEfficiency, breakdown: schema.Breakdown,
           out_base: Path) -> Tuple[Path, Path]:
    """Build the figure and save a vector PDF + 300-dpi PNG under ``out_base``."""
    with paper_style(width=_FIG_WIDTH_INCHES, height=_FIG_HEIGHT_INCHES,
                     font_size=7):
        fig = _assemble(device, strong, weak, breakdown)
        try:
            return save_paper_figure(fig, out_base)
        finally:
            plt.close(fig)
