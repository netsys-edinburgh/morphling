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

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "opt-125m",
    "opt-1.3b",
    "llama2-7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
]

METHODS = ["cleave", "dtfm", "asteroid", "confident", "alpa"]

DEVICE_COUNTS = [64, 128, 256, 512, 1024, 2048]

MODEL_PARAM_COUNTS = {
    "opt-125m": 125e6,
    "opt-1.3b": 1.3e9,
    "llama2-7b": 7e9,
    "opt-13b": 13e9,
    "opt-30b": 30e9,
    "opt-66b": 66e9,
}

MODEL_PARAM_TICK_LABELS = {
    "opt-125m": "125M",
    "opt-1.3b": "1.3B",
    "llama2-7b": "7B",
    "opt-13b": "13B",
    "opt-30b": "30B",
    "opt-66b": "66B",
}

WONG_PALETTE = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

METHOD_COLORS = {
    "cleave": WONG_PALETTE[0],
    "dtfm": WONG_PALETTE[1],
    "asteroid": WONG_PALETTE[2],
    "confident": WONG_PALETTE[3],
    "alpa": WONG_PALETTE[5],
}

METHOD_MARKERS = {
    "cleave": "o",
    "dtfm": "s",
    "asteroid": "^",
    "confident": "D",
    "alpa": "v",
}

COMPONENT_KEYS = ["compute_ms", "network_ms", "allreduce_ms", "bubble_ms"]
COMPONENT_COLORS = {
    "compute_ms": WONG_PALETTE[5],
    "network_ms": WONG_PALETTE[2],
    "allreduce_ms": WONG_PALETTE[1],
    "bubble_ms": WONG_PALETTE[7],
}
COMPONENT_LABELS = {
    "compute_ms": "compute",
    "network_ms": "network",
    "allreduce_ms": "allreduce",
    "bubble_ms": "bubble",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def index_calibrated_results(
    projections: dict[str, Any],
) -> dict[tuple[str, str, int], dict[str, float]]:
    index: dict[tuple[str, str, int], dict[str, float]] = {}
    raw_results = projections.get("results", [])
    if not isinstance(raw_results, list):
        return index

    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        method = entry.get("method")
        n = entry.get("num_devices")
        calibrated = entry.get("calibrated")
        if not isinstance(model, str) or not isinstance(method, str):
            continue
        if not isinstance(n, int) or not isinstance(calibrated, dict):
            continue

        component_values: dict[str, float] = {}
        total_ms = safe_float(calibrated.get("total_ms"))
        if total_ms is None or total_ms <= 0:
            continue
        component_values["total_ms"] = total_ms

        for key in COMPONENT_KEYS:
            v = safe_float(calibrated.get(key))
            component_values[key] = v if v is not None else 0.0

        index[(model, method, n)] = component_values

    return index


def calibration_n_devs(calibration: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    per_method = calibration.get("per_method", {})
    if not isinstance(per_method, dict):
        return out
    for method, info in per_method.items():
        if not isinstance(method, str) or not isinstance(info, dict):
            continue
        n = info.get("calibration_n_devs")
        if isinstance(n, int):
            out[method] = n
    return out


def save_pdf_png(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_stem.parent / f"{out_stem.name}.pdf"
    png_path = out_stem.parent / f"{out_stem.name}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


def plot_device_scaling(
    data: dict[tuple[str, str, int], dict[str, float]],
    calibration_points: dict[str, int],
    fig_dir: Path,
) -> None:
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        plotted_any = False

        for method in METHODS:
            points: list[tuple[int, float]] = []
            for n in DEVICE_COUNTS:
                record = data.get((model, method, n))
                if record is None:
                    continue
                points.append((n, record["total_ms"]))

            if not points:
                continue

            plotted_any = True
            points.sort(key=lambda x: x[0])
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] for p in points], dtype=float)

            cal_n = calibration_points.get(method)
            cal_mask = np.array([x == cal_n for x in xs], dtype=bool)
            proj_mask = ~cal_mask

            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method],
                label=method,
                linewidth=1.3,
            )

            if np.any(proj_mask):
                ax.scatter(
                    xs[proj_mask],
                    ys[proj_mask],
                    marker=METHOD_MARKERS[method],
                    facecolors="none",
                    edgecolors=METHOD_COLORS[method],
                    linewidths=1.1,
                    s=28,
                )

            if np.any(cal_mask):
                ax.scatter(
                    xs[cal_mask],
                    ys[cal_mask],
                    marker=METHOD_MARKERS[method],
                    facecolors=METHOD_COLORS[method],
                    edgecolors=METHOD_COLORS[method],
                    linewidths=1.0,
                    s=30,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(DEVICE_COUNTS)
        ax.set_xticklabels([str(v) for v in DEVICE_COUNTS])
        ax.set_xlabel("Number of devices")
        ax.set_ylabel("Batch runtime (ms)")
        ax.set_title(f"Device Scaling — {model}")
        ax.grid(True, which="both")

        if plotted_any:
            ax.legend(ncol=2, frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        save_pdf_png(fig, fig_dir / f"device_scaling_{model}")


def plot_model_scaling(
    data: dict[tuple[str, str, int], dict[str, float]],
    fig_dir: Path,
) -> None:
    x_ticks = [MODEL_PARAM_COUNTS[m] for m in MODELS]
    x_labels = [MODEL_PARAM_TICK_LABELS[m] for m in MODELS]

    for n in DEVICE_COUNTS:
        fig, ax = plt.subplots(figsize=(3.5, 2.7))
        plotted_any = False

        for method in METHODS:
            points: list[tuple[float, float]] = []
            for model in MODELS:
                record = data.get((model, method, n))
                if record is None:
                    continue
                points.append((MODEL_PARAM_COUNTS[model], record["total_ms"]))

            if not points:
                continue

            plotted_any = True
            points.sort(key=lambda x: x[0])
            xs = np.array([p[0] for p in points], dtype=float)
            ys = np.array([p[1] for p in points], dtype=float)
            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4,
                label=method,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Model parameter count")
        ax.set_ylabel("Batch runtime (ms)")
        ax.set_title(f"Model Scaling — N={n}")
        ax.grid(True, which="both")

        if plotted_any:
            ax.legend(ncol=2, frameon=False)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        save_pdf_png(fig, fig_dir / f"model_scaling_n{n}")


def plot_normalized_to_cleave(
    data: dict[tuple[str, str, int], dict[str, float]],
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4), sharex=False)
    flat_axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = flat_axes[idx]
        plotted_any = False

        for method in METHODS:
            xs: list[int] = []
            ys: list[float] = []
            for n in DEVICE_COUNTS:
                cleave_rec = data.get((model, "cleave", n))
                method_rec = data.get((model, method, n))
                if cleave_rec is None or method_rec is None:
                    continue
                cleave_total = safe_float(cleave_rec.get("total_ms"))
                method_total = safe_float(method_rec.get("total_ms"))
                if cleave_total is None or method_total is None:
                    continue
                if cleave_total <= 0:
                    continue
                xs.append(n)
                ys.append(method_total / cleave_total)

            if not xs:
                continue

            plotted_any = True
            ax.plot(
                xs,
                ys,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=3.5,
                label=method,
            )

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(DEVICE_COUNTS)
        ax.set_xticklabels([str(v) for v in DEVICE_COUNTS])
        ax.set_xlabel("Device count")
        ax.set_ylabel("method_time / cleave_time")
        ax.set_title(model)
        ax.grid(True, which="both")

        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

    for idx in range(len(MODELS), len(flat_axes)):
        flat_axes[idx].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    save_pdf_png(fig, fig_dir / "normalized_to_cleave")


def find_closest_device_count(
    data: dict[tuple[str, str, int], dict[str, float]],
    model: str,
    target_n: int,
) -> int | None:
    available = sorted({n for (m, _, n) in data.keys() if m == model})
    if not available:
        return None
    return min(available, key=lambda n: abs(n - target_n))


def plot_component_breakdown(
    data: dict[tuple[str, str, int], dict[str, float]],
    fig_dir: Path,
) -> None:
    targets = [(64, "opt-125m"), (512, "opt-13b"), (2048, "opt-66b")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=False)

    for ax, (target_n, model) in zip(axes, targets):
        chosen_n = find_closest_device_count(data, model, target_n)
        if chosen_n is None:
            ax.axis("off")
            continue

        x = np.arange(len(METHODS))
        bottom = np.zeros(len(METHODS), dtype=float)

        for component in COMPONENT_KEYS:
            vals = []
            for method in METHODS:
                rec = data.get((model, method, chosen_n))
                if rec is None:
                    vals.append(0.0)
                    continue
                v = safe_float(rec.get(component))
                vals.append(v if v is not None and v >= 0 else 0.0)

            vals_arr = np.array(vals, dtype=float)
            ax.bar(
                x,
                vals_arr,
                bottom=bottom,
                width=0.72,
                color=COMPONENT_COLORS[component],
                label=COMPONENT_LABELS[component],
                edgecolor="black",
                linewidth=0.35,
            )
            bottom += vals_arr

        title = f"{model}, N={chosen_n}"
        if chosen_n != target_n:
            title = f"{model}, N={chosen_n} (closest to {target_n})"
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=25, ha="right")
        ax.set_ylabel("Runtime (ms)")
        ax.grid(True, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 1.10),
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    save_pdf_png(fig, fig_dir / "component_breakdown")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    projections_path = (
        repo_root / "results" / "analytical_scaling" / "projections.json"
    )
    calibration_path = (
        repo_root / "results" / "analytical_scaling" / "calibration.json"
    )
    fig_dir = repo_root / "results" / "analytical_scaling" / "figures"

    configure_style()

    projections = load_json(projections_path)
    calibration = load_json(calibration_path)

    indexed = index_calibrated_results(projections)
    calibration_points = calibration_n_devs(calibration)

    plot_device_scaling(indexed, calibration_points, fig_dir)
    plot_model_scaling(indexed, fig_dir)
    plot_normalized_to_cleave(indexed, fig_dir)
    plot_component_breakdown(indexed, fig_dir)


if __name__ == "__main__":
    main()
