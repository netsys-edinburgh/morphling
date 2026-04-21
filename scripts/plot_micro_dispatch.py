#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUnknownLambdaType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportMissingTypeArgument=false, reportUnannotatedClassAttribute=false, reportRedeclaration=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WONG = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot micro dispatch overhead figure"
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=2.7)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _style(width: float, height: float) -> dict[str, Any]:
    return {
        "figure.figsize": (width, height),
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _cdf(samples: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.array([0.0]), np.array([1.0])
    xs = np.array(sorted(float(v) for v in samples), dtype=float)
    ys = (np.arange(xs.size, dtype=float) + 1.0) / float(xs.size)
    return xs, ys


def _save(fig: plt.Figure, out_pdf: Path) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_pdf}")
    print(f"[OK] wrote {out_png}")


def main() -> int:
    args = _parse_args()
    root = _repo_root()
    input_json = (
        args.input_json
        if args.input_json is not None
        else root / "results" / "microbenchmarks" / "dispatch_overhead.json"
    )
    output_pdf = (
        args.output_pdf
        if args.output_pdf is not None
        else root / "figures" / "evaluation" / "micro_dispatch.pdf"
    )

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    modes = payload.get("modes", {})
    eager = modes.get("eager", {}) if isinstance(modes, dict) else {}
    bp = modes.get("backpressure", {}) if isinstance(modes, dict) else {}

    eager_timeline = eager.get("queue_occupancy_timeline", [])
    bp_timeline = bp.get("queue_occupancy_timeline", [])

    eager_x = [float(pt.get("t_ms", 0.0)) for pt in eager_timeline]
    eager_y = [float(pt.get("total_queue_mb", 0.0)) for pt in eager_timeline]
    bp_x = [float(pt.get("t_ms", 0.0)) for pt in bp_timeline]
    bp_y = [float(pt.get("total_queue_mb", 0.0)) for pt in bp_timeline]

    eager_samples = eager.get("commit_delay_ms", {}).get("samples", [])
    bp_samples = bp.get("commit_delay_ms", {}).get("samples", [])
    cdf_x_eager, cdf_y_eager = _cdf([float(v) for v in eager_samples])
    cdf_x_bp, cdf_y_bp = _cdf([float(v) for v in bp_samples])

    eager_runtime = float(eager.get("batch_runtime_ms", 0.0))
    bp_runtime = float(bp.get("batch_runtime_ms", 0.0))

    with matplotlib.rc_context(_style(width=float(args.width), height=float(args.height))):
        fig, axes = plt.subplots(1, 3)

        ax0 = axes[0]
        ax0.plot(eager_x, eager_y, color="#D55E00", label="Eager")
        ax0.plot(bp_x, bp_y, color="#0072B2", label="Backpressure")
        ax0.set_xlabel("Time (ms)")
        ax0.set_ylabel("Queue occupancy (MB)")
        ax0.set_title("A. Queue occupancy")
        ax0.grid(True, axis="both")
        ax0.legend(loc="best", frameon=False)

        ax1 = axes[1]
        ax1.plot(cdf_x_eager, cdf_y_eager, color="#D55E00", label="Eager")
        ax1.plot(cdf_x_bp, cdf_y_bp, color="#0072B2", label="Backpressure")
        ax1.set_xlabel("Pending→Running latency (ms)")
        ax1.set_ylabel("CDF")
        ax1.set_title("B. Commit latency CDF")
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, axis="both")

        ax2 = axes[2]
        bars = ax2.bar(
            ["Eager", "Backpressure"],
            [eager_runtime, bp_runtime],
            color=["#D55E00", "#0072B2"],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, [eager_runtime, bp_runtime]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax2.set_ylabel("Per-batch runtime (ms)")
        ax2.set_title("C. Batch runtime")
        ax2.grid(True, axis="y")

        _save(fig, output_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
