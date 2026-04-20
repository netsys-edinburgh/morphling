#!/usr/bin/env python3
# pyright: reportMissingImports=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportAny=false
# pyright: reportExplicitAny=false
# pyright: reportImplicitStringConcatenation=false
# pyright: reportUnusedCallResult=false

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(
    0,
    "/home/xly/AgentSkillsArxiv/skills/conference-plot/scripts",
)
from plot_utils import (  # type: ignore[reportMissingImports]
    paper_style,
    save_dual_output,
)


MODEL_SOURCES: list[tuple[str, Path]] = [
    (
        "OPT-125M",
        Path(
            "/home/xly/DeviceEmulator/results/sweeps/model_size/"
            "points/000_opt-125m/comparison/summary.json"
        ),
    ),
    (
        "OPT-1.3B",
        Path(
            "/home/xly/DeviceEmulator/results/sweeps/model_size/"
            "points/001_opt-1p3b/comparison/summary.json"
        ),
    ),
    (
        "OPT-13B",
        Path(
            "/home/xly/DeviceEmulator/results/sweeps/model_size/"
            "points/002_opt-13b/comparison/summary.json"
        ),
    ),
    (
        "Llama2-7B",
        Path(
            "/home/xly/DeviceEmulator/results/sweeps/model_size/"
            "points/003_llama2-7b/comparison/summary.json"
        ),
    ),
    (
        "Llama2-13B",
        Path(
            "/home/xly/DeviceEmulator/results/sweeps/model_size/"
            "points/004_llama2-13b/comparison/summary.json"
        ),
    ),
]

BASELINE_ORDER = ["dtfm", "asteroid", "confident", "alpa", "cleave"]
BASELINE_LABELS = {
    "cleave": "Wasp",
    "dtfm": "DTFM",
    "asteroid": "Asteroid",
    "confident": "Confident",
    "alpa": "Alpa",
}


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"[WARN] Missing summary: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None


def get_runtime_ms(summary: dict[str, Any], baseline: str) -> float | None:
    baselines = summary.get("baselines")
    if not isinstance(baselines, dict):
        return None
    stats = baselines.get(baseline)
    if not isinstance(stats, dict):
        return None
    runtime = stats.get("total_runtime_ms")
    try:
        return float(runtime)
    except (TypeError, ValueError):
        return None


def format_runtime(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def collect_normalized_data_analytical() -> tuple[
    dict[str, dict[str, float]],
    dict[str, float],
    list[str],
]:
    """Load from per-model sweep summary JSONs (analytical)."""
    normalized: dict[str, dict[str, float]] = {}
    cleave_seconds: dict[str, float] = {}

    for model, path in MODEL_SOURCES:
        normalized[model] = {b: np.nan for b in BASELINE_ORDER}
        summary = load_summary(path)
        if summary is None:
            continue

        cleave_ms = get_runtime_ms(summary, "cleave")
        if cleave_ms is None or cleave_ms <= 0:
            print(f"[WARN] Missing Wasp runtime for {model}: {path}")
            continue

        cleave_seconds[model] = cleave_ms / 1000.0
        for baseline in BASELINE_ORDER:
            runtime_ms = get_runtime_ms(summary, baseline)
            if runtime_ms is None:
                continue
            normalized[model][baseline] = runtime_ms / cleave_ms

    models = [m for m, _ in MODEL_SOURCES]
    return normalized, cleave_seconds, models


def collect_normalized_data_vtime(
    summary_path: Path,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float],
    list[str],
]:
    """Load from VTIME summary (aggregate or paper_summary).

    Supports two formats:
    1. aggregate_vtime_results.py output:
       {"models": {"OPT-125M": {"baselines": {"cleave":
         {"total_runtime_ms": ...}}}}}
    2. paper_summary.json:
       {"results": {"cleave": {"total_ms": ...}}}
    """
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    normalized: dict[str, dict[str, float]] = {}
    cleave_seconds: dict[str, float] = {}
    models: list[str] = []

    # Format 1: aggregate output with "models" key
    if "models" in data:
        for label, mdata in data["models"].items():
            baselines = mdata.get("baselines", {})
            cleave_ms = None
            entry = baselines.get("cleave", {})
            val = entry.get("total_runtime_ms")
            if val is not None:
                cleave_ms = float(val)

            if cleave_ms is None or cleave_ms <= 0:
                print(f"[WARN] Missing Wasp VTIME for {label}")
                continue

            models.append(label)
            cleave_seconds[label] = cleave_ms / 1000.0
            normalized[label] = {}
            for b in BASELINE_ORDER:
                bdata = baselines.get(b, {})
                rt = bdata.get("total_runtime_ms")
                if rt is None:
                    normalized[label][b] = np.nan
                else:
                    normalized[label][b] = float(rt) / cleave_ms
        return normalized, cleave_seconds, models

    # Format 2: paper_summary.json with "results" key
    results = data.get("results", {})
    cleave_entry = results.get("cleave", {})
    cleave_ms = cleave_entry.get("total_ms")
    if cleave_ms is None:
        raise ValueError(f"No cleave total_ms in {summary_path}")
    cleave_ms = float(cleave_ms)

    raw_label = data.get("model", "OPT-125M")
    label = raw_label.split(" (")[0]
    models.append(label)
    cleave_seconds[label] = cleave_ms / 1000.0
    normalized[label] = {}
    for b in BASELINE_ORDER:
        entry = results.get(b, {})
        rt = entry.get("total_ms")
        if rt is None:
            normalized[label][b] = np.nan
        else:
            normalized[label][b] = float(rt) / cleave_ms

    return normalized, cleave_seconds, models

    return [
        label for label, _ in sorted(scores.items(), key=_key, reverse=True)
    ]


def sort_models_and_baselines(
    normalized: dict[str, dict[str, float]],
) -> tuple[list[str], list[str]]:
    model_scores: dict[str, float] = {}
    for model, _ in MODEL_SOURCES:
        vals = np.array(
            [normalized[model][baseline] for baseline in BASELINE_ORDER],
            dtype=float,
        )
        finite_vals = vals[np.isfinite(vals)]
        model_scores[model] = (
            float(np.nanmean(finite_vals)) if finite_vals.size else -np.inf
        )

    ordered_models = _descending_order_by_score(model_scores)

    baseline_scores: dict[str, float] = {}
    for baseline in BASELINE_ORDER:
        vals = np.array(
            [normalized[model][baseline] for model in ordered_models],
            dtype=float,
        )
        finite_vals = vals[np.isfinite(vals)]
        baseline_scores[baseline] = (
            float(np.nanmean(finite_vals)) if finite_vals.size else -np.inf
        )

    ordered_baselines = _descending_order_by_score(baseline_scores)
    return ordered_models, ordered_baselines


BASELINE_COLORS = {
    "dtfm": "#FFFFFF",
    "asteroid": "#D9D9D9",
    "confident": "#969696",
    "alpa": "#525252",
    "cleave": "#000000",
}


def plot_e2e_latency(
    vtime_summary: Path | None = None,
) -> None:
    if vtime_summary is not None:
        normalized, cleave_seconds, models = collect_normalized_data_vtime(
            vtime_summary
        )
    else:
        normalized, cleave_seconds, models = (
            collect_normalized_data_analytical()
        )

    out_pdf = Path("figures/comparison/fig_e2e_latency.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    with paper_style(width=7.0, height=2.5):
        fig = cast(Any, plt.figure())
        ax = fig.add_subplot(111)
        x = np.arange(len(models))
        width = 0.15
        offsets = (
            np.arange(len(BASELINE_ORDER)) - (len(BASELINE_ORDER) - 1) / 2.0
        ) * width

        legend_handles: list[Patch] = []
        positive_values: list[float] = []

        for idx, baseline in enumerate(BASELINE_ORDER):
            values = np.array(
                [normalized.get(m, {}).get(baseline, np.nan) for m in models],
                dtype=float,
            )
            color = BASELINE_COLORS[baseline]
            bars = ax.bar(
                x + offsets[idx],
                values,
                width=width,
                color=color,
                edgecolor="black",
                linewidth=0.4,
            )
            legend_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.4,
                    label=BASELINE_LABELS[baseline],
                )
            )

            positive_values.extend(values[np.isfinite(values) & (values > 0)])

            if baseline != "cleave":
                continue

            for rect, model in zip(bars, models):
                val = normalized.get(model, {}).get(baseline, np.nan)
                secs = cleave_seconds.get(model)
                if not np.isfinite(val) or secs is None:
                    continue
                ax.annotate(
                    format_runtime(secs),
                    xy=(
                        rect.get_x() + rect.get_width() / 2.0,
                        val,
                    ),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel("Model")
        ax.set_ylabel("Normalized runtime")
        # ax.set_yscale("log")
        ax.set_ylim(0,30)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(
            1.0,
            color="gray",
            linestyle="--",
            linewidth=0.5,
        )
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=5,
            frameon=False,
        )

        if positive_values:
            low = min(positive_values) * 0.5
            high = max(positive_values) * 2.0
            ax.set_ylim(bottom=min(0.5, low), top=high)

        fig.subplots_adjust(top=0.82)
        save_dual_output(fig, out_pdf, None)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot e2e latency (fig:eval-e2e-lat)"
    )
    p.add_argument(
        "--vtime-summary",
        default=None,
        type=str,
        help=(
            "Path to VTIME summary JSON. "
            "If omitted, uses analytical sweep data."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vtime_path = Path(args.vtime_summary) if args.vtime_summary else None
    plot_e2e_latency(vtime_summary=vtime_path)


if __name__ == "__main__":
    main()
