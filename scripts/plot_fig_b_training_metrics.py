#!/usr/bin/env python3
"""Figure B: Grouped bar chart of training metrics.

Usage:
  python3 scripts/plot_fig_b_training_metrics.py \
      --data results/paper_data.json \
      --output figures/fig_b_training_metrics.pdf
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WONG = {
    "orange": "#E69F00",
    "blue": "#0072B2",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "black": "#000000",
}

plt.rcParams.update({
    "figure.figsize": (3.33, 2.5),
    "font.size": 8,
    "font.family": "serif",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COLOR_NO_CTRL = "#999999"  # gray
COLOR_WEAVER = WONG["blue"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure B: Training metrics bar chart"
    )
    parser.add_argument(
        "--data", default="results/paper_data.json"
    )
    parser.add_argument(
        "--output", default="figures/fig_b_training_metrics.pdf"
    )
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    no = data.get("without_ctrl", {})
    we = data.get("with_ctrl", {})

    # Extract metrics
    metrics = {
        "Token\nThroughput": {
            "no_ctrl": no.get("tokens_per_sec_mean", 1.0),
            "no_ctrl_std": no.get("tokens_per_sec_mean_std", 0),
            "weaver": we.get("tokens_per_sec_mean", 1.0),
            "weaver_std": we.get("tokens_per_sec_mean_std", 0),
            "err_type": "std",
        },
        "Achieved\nFLOPS": {
            "no_ctrl": no.get("gflops_mean", 1.0) or 1.0,
            "no_ctrl_std": no.get("gflops_mean_std", 0),
            "weaver": we.get("gflops_mean", 1.0) or 1.0,
            "weaver_std": we.get("gflops_mean_std", 0),
            "err_type": "std",
        },
        "Per-Step\nLatency": {
            "no_ctrl": no.get("step_time_ms_mean", 1.0),
            "no_ctrl_min": no.get("step_time_ms_run_min", 0),
            "no_ctrl_max": no.get("step_time_ms_run_max", 0),
            "weaver": we.get("step_time_ms_mean", 1.0),
            "weaver_min": we.get("step_time_ms_run_min", 0),
            "weaver_max": we.get("step_time_ms_run_max", 0),
            "err_type": "minmax",
        },
    }

    labels = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(3.33, 2.5))

    # Normalize to No Control = 1.0
    no_vals = []
    we_vals = []
    no_errs = []
    we_errs = []

    for label in labels:
        m = metrics[label]
        base = m["no_ctrl"] if m["no_ctrl"] != 0 else 1.0
        no_norm = 1.0
        we_norm = m["weaver"] / base

        no_vals.append(no_norm)
        we_vals.append(we_norm)

        if m["err_type"] == "std":
            no_errs.append(m.get("no_ctrl_std", 0) / base)
            we_errs.append(m.get("weaver_std", 0) / base)
        else:
            # min-max as asymmetric error bars
            no_low = (m["no_ctrl"] - m.get("no_ctrl_min", m["no_ctrl"])) / base
            no_high = (m.get("no_ctrl_max", m["no_ctrl"]) - m["no_ctrl"]) / base
            we_low = (m["weaver"] - m.get("weaver_min", m["weaver"])) / base
            we_high = (m.get("weaver_max", m["weaver"]) - m["weaver"]) / base
            no_errs.append(max(no_low, no_high))
            we_errs.append(max(we_low, we_high))

    bars_no = ax.bar(
        x - width / 2, no_vals, width,
        yerr=no_errs, capsize=2,
        label="No Control", color=COLOR_NO_CTRL,
        edgecolor="white", linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )
    bars_we = ax.bar(
        x + width / 2, we_vals, width,
        yerr=we_errs, capsize=2,
        label="Weaver", color=COLOR_WEAVER,
        edgecolor="white", linewidth=0.3,
        error_kw={"linewidth": 0.5},
    )

    # Percentage delta annotations
    for i, (nv, wv) in enumerate(zip(no_vals, we_vals)):
        delta = (wv - nv) / nv * 100
        sign = "+" if delta >= 0 else ""
        ax.text(
            x[i] + width / 2, wv + we_errs[i] + 0.02,
            f"{sign}{delta:.1f}%",
            ha="center", va="bottom", fontsize=6,
            color=WONG["green"] if delta >= 0 else WONG["vermillion"],
        )

    # Baseline reference line
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_ylabel("Normalized Metric (baseline = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(no_vals), max(we_vals)) * 1.25)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf")
    print(f"Saved {args.output}")
    fig.savefig(args.output.replace(".pdf", ".png"), format="png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
