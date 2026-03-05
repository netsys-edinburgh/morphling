#!/usr/bin/env python3
"""Figure A: SM allocation timeline comparing No Control vs Weaver.

Usage:
  python3 scripts/plot_fig_a_sm_timeline.py \
      --trace-no-ctrl data/ldpc_trace_without_ctrl.csv \
      --trace-with-ctrl data/ldpc_trace_with_ctrl.csv \
      --violations-no-ctrl results/without_ctrl/run_0/violations.json \
      --violations-with-ctrl results/with_ctrl/run_0/violations.json \
      --output figures/fig_a_sm_timeline.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Wong colorblind-safe palette
WONG = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

# Paper RC params (matching eval_greenctx_training.py)
plt.rcParams.update({
    "figure.figsize": (3.33, 3.0),
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 0.6,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load_violations(path: str) -> set:
    """Load violation slot indices from JSON."""
    if not path or not os.path.exists(path):
        return set()
    with open(path) as f:
        data = json.load(f)
    return {
        v["step"]
        for v in data.get("per_step_violations", [])
        if v.get("violations", 0) > 0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot SM allocation timeline (Figure A)"
    )
    parser.add_argument(
        "--trace-no-ctrl",
        default="data/ldpc_trace_without_ctrl.csv",
    )
    parser.add_argument(
        "--trace-with-ctrl",
        default="data/ldpc_trace_with_ctrl.csv",
    )
    parser.add_argument("--violations-no-ctrl", default="")
    parser.add_argument("--violations-with-ctrl", default="")
    parser.add_argument(
        "--output", default="figures/fig_a_sm_timeline.pdf"
    )
    parser.add_argument(
        "--max-sms", type=int, default=48,
        help="Y-axis max",
    )
    parser.add_argument(
        "--window-start", type=int, default=0,
        help="Start slot index for window (0=full trace)",
    )
    parser.add_argument(
        "--window-size", type=int, default=0,
        help="Window size in slots (0=full trace)",
    )
    args = parser.parse_args()

    # Load traces
    df_no = pd.read_csv(args.trace_no_ctrl)
    df_with = pd.read_csv(args.trace_with_ctrl)

    # Apply window
    if args.window_size > 0:
        s, e = args.window_start, args.window_start + args.window_size
        df_no = df_no.iloc[s:e].reset_index(drop=True)
        df_with = df_with.iloc[s:e].reset_index(drop=True)

    # Convert timestamps to ms
    t0_no = df_no["time_slot_sched_ns"].iloc[0]
    t0_with = df_with["time_slot_sched_ns"].iloc[0]
    x_no = (df_no["time_slot_sched_ns"] - t0_no) / 1e6
    x_with = (df_with["time_slot_sched_ns"] - t0_with) / 1e6
    y_no = df_no["sm_count"]
    y_with = df_with["sm_count"]

    # Load violations
    viol_no = load_violations(args.violations_no_ctrl)
    viol_with = load_violations(args.violations_with_ctrl)

    # Create figure
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(3.33, 3.0)
    )

    # TOP: No Control (gray/orange family)
    ax_top.fill_between(
        x_no, y_no, alpha=0.3, color=WONG["orange"],
        step="post",
    )
    ax_top.step(
        x_no, y_no, where="post",
        color=WONG["orange"], linewidth=0.6,
    )

    # Violation ticks
    if viol_no:
        viol_mask = df_no.index.isin(viol_no)
        if viol_mask.any():
            viol_x = x_no[viol_mask]
            for vx in viol_x:
                ax_top.axvline(
                    vx, color=WONG["vermillion"],
                    alpha=0.4, linewidth=0.3,
                )

    ax_top.set_ylabel("RAN SMs")
    ax_top.set_ylim(0, args.max_sms)
    ax_top.set_title("No Control (passive)", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # Annotation: SM swap count
    n_swaps_no = (df_no["sm_count"].diff() != 0).sum() - 1
    ax_top.text(
        0.98, 0.92,
        f"{n_swaps_no:,} swaps",
        transform=ax_top.transAxes,
        fontsize=6, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )

    # BOTTOM: Weaver (blue/green family)
    ax_bot.fill_between(
        x_with, y_with, alpha=0.3, color=WONG["blue"],
        step="post",
    )
    ax_bot.step(
        x_with, y_with, where="post",
        color=WONG["blue"], linewidth=0.6,
    )

    # Violation ticks
    if viol_with:
        viol_mask = df_with.index.isin(viol_with)
        if viol_mask.any():
            viol_x = x_with[viol_mask]
            for vx in viol_x:
                ax_bot.axvline(
                    vx, color=WONG["vermillion"],
                    alpha=0.4, linewidth=0.3,
                )

    ax_bot.set_ylabel("RAN SMs")
    ax_bot.set_xlabel("Time (ms)")
    ax_bot.set_ylim(0, args.max_sms)
    ax_bot.set_title("Weaver (active control)", fontsize=8)
    ax_bot.grid(True, alpha=0.3)

    # Annotation: SM swap count + hysteresis note
    n_swaps_with = (df_with["sm_count"].diff() != 0).sum() - 1
    reduction = (1 - n_swaps_with / max(n_swaps_no, 1)) * 100
    ax_bot.text(
        0.98, 0.92,
        f"{n_swaps_with:,} swaps ({reduction:.0f}% fewer)",
        transform=ax_bot.transAxes,
        fontsize=6, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
    )

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf")
    print(f"Saved {args.output}")

    # Also save PNG for quick preview
    png_path = args.output.replace(".pdf", ".png")
    fig.savefig(png_path, format="png", dpi=300)
    print(f"Saved {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
