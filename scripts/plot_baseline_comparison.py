#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINE_ORDER = ["cleave", "dtfm", "asteroid", "confident", "alpa"]
BASELINE_LABELS = {
    "cleave": "CLEAVE",
    "dtfm": "DTFM",
    "asteroid": "Asteroid",
    "confident": "Confident",
    "alpa": "Alpa",
}

BASELINE_COLORS = {
    "cleave": "#0072B2",
    "dtfm": "#D55E00",
    "asteroid": "#009E73",
    "confident": "#CC79A7",
    "alpa": "#F0E442",
}

BASELINE_MARKERS = {
    "cleave": "o",
    "dtfm": "s",
    "asteroid": "^",
    "confident": "D",
    "alpa": "v",
}

MODEL_ORDER = [
    "OPT-125M",
    "OPT-1.3B",
    "OPT-13B",
    "Llama2-7B",
    "Llama2-13B",
]

PHASE_KEYS = [
    ("compute_time_ms", "compute", "#0072B2"),
    ("network_time_ms", "network", "#56B4E9"),
    ("bubble_time_ms", "bubble", "#E69F00"),
    ("allreduce_time_ms", "allreduce", "#009E73"),
]


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "grid.color": "#CCCCCC",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / filename
    png_path = output_dir / filename.replace(".pdf", ".png")
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def canonical_baseline_name(name: str) -> str:
    return str(name).strip().lower()


def extract_runtime_ms(baseline_stats: dict[str, Any]) -> float | None:
    for key in (
        "total_runtime_ms",
        "runtime_ms",
        "per_batch_runtime_ms",
        "batch_runtime_ms",
        "normalized_runtime",
        "normalized_total_runtime",
    ):
        if key in baseline_stats:
            try:
                return float(baseline_stats[key])
            except (TypeError, ValueError):
                return None
    return None


def parse_sweep_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return [x for x in data["results"] if isinstance(x, dict)]
        if isinstance(data.get("sweep_results"), list):
            return [x for x in data["sweep_results"] if isinstance(x, dict)]
    return []


def get_param_value(record: dict[str, Any]) -> Any:
    for key in ("param_value", "value", "model", "model_name"):
        if key in record:
            return record[key]
    return None


def get_baselines(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    obj = record.get("baselines", {})
    if not isinstance(obj, dict):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for name, stats in obj.items():
        if isinstance(stats, dict):
            parsed[canonical_baseline_name(name)] = stats
        elif isinstance(stats, (int, float)):
            parsed[canonical_baseline_name(name)] = {
                "total_runtime_ms": float(stats)
            }
    return parsed


def find_first_existing_json(base_dir: Path, stems: list[str]) -> Path | None:
    for stem in stems:
        candidate = base_dir / f"{stem}.json"
        if candidate.exists():
            return candidate
    return None


def load_sweep_data(
    sweep_dir: Path, stems: list[str]
) -> list[dict[str, Any]] | None:
    path = find_first_existing_json(sweep_dir, stems)
    if path is None:
        return None
    data = load_json(path)
    if data is None:
        return None
    return parse_sweep_records(data)


def plot_normalized_runtime_from_sweep(
    sweep_dir: Path,
    output_dir: Path,
) -> bool:
    records = load_sweep_data(
        sweep_dir,
        ["model", "models", "model_name", "model_size", "model_sweep"],
    )
    if not records:
        warn("Missing model sweep; skipping fig_normalized_runtime.pdf")
        return False

    normalized: dict[str, dict[str, float]] = {m: {} for m in MODEL_ORDER}
    for rec in records:
        model_raw = str(get_param_value(rec) or "")
        model = normalize_model_name(model_raw)
        if model not in normalized:
            continue

        baselines = get_baselines(rec)
        cleave = baselines.get("cleave")
        cleave_runtime = extract_runtime_ms(cleave or {})
        if cleave_runtime is None or cleave_runtime <= 0:
            continue

        for baseline in BASELINE_ORDER:
            stats = baselines.get(baseline)
            runtime = extract_runtime_ms(stats or {})
            if runtime is None:
                continue
            normalized[model][baseline] = runtime / cleave_runtime

    fig = plt.figure(figsize=(7.0, 2.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.14

    plotted_any = False
    for idx, baseline in enumerate(BASELINE_ORDER):
        vals = [normalized[m].get(baseline) for m in MODEL_ORDER]
        if not any(v is not None for v in vals):
            continue
        y = [v if v is not None else np.nan for v in vals]
        plt.bar(
            x + (idx - 2) * width,
            y,
            width=width,
            color=BASELINE_COLORS[baseline],
            label=BASELINE_LABELS[baseline],
            edgecolor="white",
            linewidth=0.4,
        )
        plotted_any = True

    if not plotted_any:
        warn(
            "No valid model runtimes found; skipping fig_normalized_runtime.pdf"
        )
        plt.close(fig)
        return False

    plt.xticks(x, MODEL_ORDER)
    plt.ylabel("Normalized per-batch runtime\n(CLEAVE = 1.0)")
    plt.xlabel("Model")
    plt.grid(True, axis="y", alpha=0.8)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=5, frameon=False
    )

    save_figure(fig, output_dir, "fig_normalized_runtime.pdf")
    return True


def plot_single_normalized_runtime(
    comparison_path: Path,
    output_dir: Path,
) -> bool:
    data = load_json(comparison_path)
    if not isinstance(data, dict):
        warn(
            "Invalid single comparison JSON; skipping fig_normalized_runtime.pdf"
        )
        return False

    baselines = get_baselines(data)
    cleave_runtime = extract_runtime_ms(baselines.get("cleave", {}))
    if cleave_runtime is None or cleave_runtime <= 0:
        warn(
            "CLEAVE runtime missing in single comparison; skipping normalized figure"
        )
        return False

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for baseline in BASELINE_ORDER:
        runtime = extract_runtime_ms(baselines.get(baseline, {}))
        if runtime is None:
            continue
        labels.append(BASELINE_LABELS[baseline])
        values.append(runtime / cleave_runtime)
        colors.append(BASELINE_COLORS[baseline])

    if not values:
        warn(
            "No baselines found in single comparison; skipping normalized figure"
        )
        return False

    fig = plt.figure(figsize=(3.3, 2.4))
    xpos = np.arange(len(labels))
    plt.bar(xpos, values, color=colors, edgecolor="white", linewidth=0.4)
    plt.xticks(xpos, labels, rotation=20, ha="right")
    plt.ylabel("Normalized per-batch runtime\n(CLEAVE = 1.0)")
    plt.grid(True, axis="y", alpha=0.8)

    save_figure(fig, output_dir, "fig_normalized_runtime.pdf")
    return True


def normalize_model_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    mapping = {
        "opt-125m": "OPT-125M",
        "facebook/opt-125m": "OPT-125M",
        "opt-1.3b": "OPT-1.3B",
        "facebook/opt-1.3b": "OPT-1.3B",
        "opt-13b": "OPT-13B",
        "facebook/opt-13b": "OPT-13B",
        "llama2-7b": "Llama2-7B",
        "meta-llama/llama-2-7b": "Llama2-7B",
        "llama-2-7b": "Llama2-7B",
        "llama2-13b": "Llama2-13B",
        "meta-llama/llama-2-13b": "Llama2-13B",
        "llama-2-13b": "Llama2-13B",
    }
    return mapping.get(key, name)


def prepare_line_series(
    records: list[dict[str, Any]],
    param_formatter=lambda v: v,
    normalize_to_first: bool = False,
) -> tuple[list[Any], dict[str, list[float]]]:
    entries: list[tuple[float, dict[str, float]]] = []
    for rec in records:
        param = get_param_value(rec)
        try:
            x_val = float(param)
        except (TypeError, ValueError):
            continue

        row: dict[str, float] = {}
        for baseline, stats in get_baselines(rec).items():
            runtime = extract_runtime_ms(stats)
            if runtime is not None:
                row[baseline] = runtime
        if row:
            entries.append((x_val, row))

    entries.sort(key=lambda x: x[0])
    if not entries:
        return [], {}

    x_values = [param_formatter(x) for x, _ in entries]
    series: dict[str, list[float]] = {}

    for baseline in BASELINE_ORDER:
        vals: list[float] = []
        for _, row in entries:
            if baseline in row:
                vals.append(float(row[baseline]))
            else:
                vals.append(np.nan)
        if (
            normalize_to_first
            and vals
            and not np.isnan(vals[0])
            and vals[0] != 0
        ):
            base = vals[0]
            vals = [v / base if not np.isnan(v) else np.nan for v in vals]
        series[baseline] = vals

    return x_values, series


def plot_line_figure(
    records: list[dict[str, Any]] | None,
    output_dir: Path,
    filename: str,
    xlabel: str,
    ylabel: str,
    xtick_labels: list[str] | None = None,
    tick_label_formatter=None,
    normalize_to_first: bool = False,
) -> bool:
    if not records:
        warn(f"Missing data; skipping {filename}")
        return False

    x_values, series = prepare_line_series(
        records,
        param_formatter=lambda v: int(v) if float(v).is_integer() else v,
        normalize_to_first=normalize_to_first,
    )
    if not x_values:
        warn(f"No valid points; skipping {filename}")
        return False

    fig = plt.figure(figsize=(3.3, 2.4))
    plotted_any = False
    x_numeric = np.arange(len(x_values))

    for baseline in BASELINE_ORDER:
        vals = series.get(baseline, [])
        if not vals or all(np.isnan(v) for v in vals):
            continue
        plt.plot(
            x_numeric,
            vals,
            marker=BASELINE_MARKERS[baseline],
            color=BASELINE_COLORS[baseline],
            label=BASELINE_LABELS[baseline],
            markersize=4,
            linewidth=1.2,
        )
        plotted_any = True

    if not plotted_any:
        warn(f"No baseline series available; skipping {filename}")
        plt.close(fig)
        return False

    if xtick_labels is not None:
        labels = xtick_labels
    elif tick_label_formatter is not None:
        labels = [str(tick_label_formatter(v)) for v in x_values]
    else:
        labels = [str(v) for v in x_values]
    plt.xticks(x_numeric, labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.8)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False
    )

    save_figure(fig, output_dir, filename)
    return True


def plot_phase_breakdown_from_summary(
    summary: dict[str, Any],
    output_dir: Path,
) -> bool:
    baselines = get_baselines(summary)
    if not baselines:
        warn("No baseline summary found; skipping fig_phase_breakdown.pdf")
        return False

    labels: list[str] = []
    stacks: dict[str, list[float]] = {phase: [] for _, phase, _ in PHASE_KEYS}

    for baseline in BASELINE_ORDER:
        stats = baselines.get(baseline)
        if not stats:
            continue
        labels.append(BASELINE_LABELS[baseline])
        for key, phase_name, _ in PHASE_KEYS:
            value = stats.get(key, 0.0)
            try:
                stacks[phase_name].append(float(value))
            except (TypeError, ValueError):
                stacks[phase_name].append(0.0)

    if not labels:
        warn("No phase entries found; skipping fig_phase_breakdown.pdf")
        return False

    fig = plt.figure(figsize=(3.3, 2.6))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for _, phase_name, color in PHASE_KEYS:
        values = np.array(stacks[phase_name], dtype=float)
        plt.bar(
            x,
            values,
            bottom=bottom,
            color=color,
            label=phase_name,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += values

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.xlabel("Baselines")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, axis="y", alpha=0.8)
    plt.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False
    )

    save_figure(fig, output_dir, "fig_phase_breakdown.pdf")
    return True


def load_summary_from_sweep_dir(sweep_dir: Path) -> dict[str, Any] | None:
    for rel in (
        "summary.json",
        "../comparison/summary.json",
        "../summary.json",
    ):
        path = (sweep_dir / rel).resolve()
        data = load_json(path)
        if isinstance(data, dict) and isinstance(data.get("baselines"), dict):
            return data
    return None


def run_sweep_mode(sweep_dir: Path, output_dir: Path) -> None:
    plot_normalized_runtime_from_sweep(sweep_dir, output_dir)

    device_records = load_sweep_data(
        sweep_dir,
        ["num_devices", "device_count", "devices", "device_scaling"],
    )
    plot_line_figure(
        device_records,
        output_dir,
        "fig_device_scaling.pdf",
        xlabel="Number of devices",
        ylabel="Per-batch runtime (ms)",
    )

    batch_records = load_sweep_data(
        sweep_dir,
        ["batch_size", "batch", "batch_scaling"],
    )
    plot_line_figure(
        batch_records,
        output_dir,
        "fig_batch_scaling.pdf",
        xlabel="Batch size",
        ylabel="Per-batch runtime (ms)",
    )

    straggler_records = load_sweep_data(
        sweep_dir,
        ["straggler_ratio", "straggler", "straggler_sensitivity"],
    )
    plot_line_figure(
        straggler_records,
        output_dir,
        "fig_straggler_sensitivity.pdf",
        xlabel="Straggler ratio",
        ylabel="Per-batch runtime (ms)",
    )

    asym_records = load_sweep_data(
        sweep_dir,
        ["dl_ul_ratio", "asymmetry", "dl_ul_asymmetry", "dlul_ratio"],
    )
    plot_line_figure(
        asym_records,
        output_dir,
        "fig_dl_ul_asymmetry.pdf",
        xlabel="DL/UL ratio",
        ylabel="Per-batch runtime (ms)",
    )

    summary = load_summary_from_sweep_dir(sweep_dir)
    if summary is None:
        warn("summary.json not found; skipping fig_phase_breakdown.pdf")
    else:
        plot_phase_breakdown_from_summary(summary, output_dir)


def run_single_comparison_mode(comparison_path: Path, output_dir: Path) -> None:
    data = load_json(comparison_path)
    if not isinstance(data, dict):
        warn("comparison file missing or invalid; no figures generated")
        return

    plot_single_normalized_runtime(comparison_path, output_dir)
    plot_phase_breakdown_from_summary(data, output_dir)

    warn("Single-comparison mode does not include sweep-based scaling figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline comparison publication figures",
    )
    parser.add_argument(
        "--sweep-dir",
        default="results/sweeps/",
        help="Directory containing sweep JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/comparison/",
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--comparison",
        default="results/comparison/summary.json",
        help="Single comparison summary.json path",
    )
    parser.add_argument(
        "--single-comparison",
        action="store_true",
        help="Generate figures from a single summary.json comparison",
    )
    return parser.parse_args()


def main() -> None:
    configure_plot_style()
    args = parse_args()

    output_dir = Path(args.output_dir)
    if args.single_comparison:
        run_single_comparison_mode(Path(args.comparison), output_dir)
    else:
        run_sweep_mode(Path(args.sweep_dir), output_dir)


if __name__ == "__main__":
    main()
