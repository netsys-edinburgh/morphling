#!/usr/bin/env python3
"""Generate publication-quality LDPC real-trace comparison figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_adapter_cls() -> Any:
    module_path = REPO_ROOT / "morphling" / "runtime" / "ldpc_trace_adapter.py"
    spec = importlib.util.spec_from_file_location("ldpc_trace_adapter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load adapter module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "LdpcTraceAdapter")


LdpcTraceAdapter = _load_adapter_cls()


WONG_PALETTE = [
    "#000000",
    "#EEBA0C",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#0000FF",
    "#FF0000",
]


@dataclass
class TraceData:
    name: str
    adapter: Any
    df: pd.DataFrame
    violations: pd.DataFrame
    inefficiencies: pd.DataFrame


def _paper_rc() -> dict[str, object]:
    return {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.3,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}'")
    vals_arr = np.asarray(pd.to_numeric(df[col], errors="coerce"), dtype=float)
    bad_mask = np.isnan(vals_arr)
    if bool(bad_mask.any()):
        bad_rows = np.flatnonzero(bad_mask)[:10].tolist()
        raise ValueError(f"Column '{col}' has non-numeric rows: {bad_rows}")
    return pd.Series(vals_arr, index=df.index)


def _load_trace(path: Path, total_sms: int, name: str) -> TraceData:
    adapter = LdpcTraceAdapter(path, total_sms=total_sms)
    df = adapter._df.copy()

    required_cols = ["decode_dur_us", "deadline_met", "sum_prb", "sum_tbs_bytes"]
    for col in required_cols:
        df[col] = _ensure_numeric(df, col)

    df["deadline_met"] = (df["deadline_met"] > 0).astype(int)
    df["row_idx"] = np.arange(len(df), dtype=int)

    target_col = None
    for candidate in ("target_sm", "target_sm_count", "target_sms"):
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        df["target_sm"] = total_sms
    else:
        df["target_sm"] = _ensure_numeric(df, target_col)

    violations = adapter.detect_violations()
    inefficiencies = adapter.detect_inefficiencies()

    return TraceData(
        name=name,
        adapter=adapter,
        df=df,
        violations=violations,
        inefficiencies=inefficiencies,
    )


def _save_dual(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def _deadline_compliance(trace_a: TraceData, trace_b: TraceData, out_dir: Path) -> None:
    labels = ["≤10", "11-40", "41-80", ">80", "overall"]
    bins = [-np.inf, 10, 40, 80, np.inf]

    def compute(df: pd.DataFrame) -> list[float]:
        bucket = pd.cut(df["sum_prb"], bins=bins, labels=labels[:-1], right=True)
        grouped = df.groupby(bucket, observed=True)["deadline_met"].mean()
        values: list[float] = []
        for lbl in labels[:-1]:
            raw = grouped.get(lbl)
            values.append(float(raw) * 100.0 if raw is not None else float("nan"))
        values.append(float(df["deadline_met"].mean()) * 100.0)
        return values

    vals_a = compute(trace_a.df)
    vals_b = compute(trace_b.df)

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        x = np.arange(len(labels), dtype=float)
        width = 0.38
        ax.bar(
            x - width / 2,
            vals_a,
            width,
            label=trace_a.name,
            color=WONG_PALETTE[5],
            edgecolor="black",
            linewidth=0.4,
            hatch="//",
        )
        ax.bar(
            x + width / 2,
            vals_b,
            width,
            label=trace_b.name,
            color=WONG_PALETTE[1],
            edgecolor="black",
            linewidth=0.4,
            hatch="\\",
        )
        ax.set_ylim(0, 105)
        ax.set_ylabel("Deadline met (%)")
        ax.set_xlabel("Workload intensity by sum_prb")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title("Deadline Compliance by Workload Bin")
        ax.legend(loc="lower left")
        _save_dual(fig, out_dir, "deadline_compliance")


def _cdf_xy(vals):
    x = np.sort(np.asarray(vals, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / float(len(x))
    return x, y


def _latency_cdf(trace_a: TraceData, trace_b: TraceData, out_dir: Path) -> None:
    data_a = pd.Series(trace_a.df["decode_dur_us"])
    data_b = pd.Series(trace_b.df["decode_dur_us"])
    x_a, y_a = _cdf_xy(data_a)
    x_b, y_b = _cdf_xy(data_b)

    q_a = np.percentile(data_a.to_numpy(dtype=float), [50, 95, 99])
    q_b = np.percentile(data_b.to_numpy(dtype=float), [50, 95, 99])

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        ax.plot(x_a, y_a, color=WONG_PALETTE[5], label=trace_a.name)
        ax.plot(x_b, y_b, color=WONG_PALETTE[1], label=trace_b.name)

        styles = ["--", ":", "-."]
        for idx, q in enumerate(q_a):
            ax.axvline(q, color=WONG_PALETTE[5], linestyle=styles[idx], alpha=0.8)
        for idx, q in enumerate(q_b):
            ax.axvline(q, color=WONG_PALETTE[1], linestyle=styles[idx], alpha=0.8)

        annotation = (
            f"{trace_a.name} p50/p95/p99: {q_a[0]:.1f}/{q_a[1]:.1f}/{q_a[2]:.1f} µs\n"
            f"{trace_b.name} p50/p95/p99: {q_b[0]:.1f}/{q_b[1]:.1f}/{q_b[2]:.1f} µs"
        )
        ax.text(
            0.03,
            0.03,
            annotation,
            transform=ax.transAxes,
            fontsize=6.5,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        ax.set_xlabel("decode_dur_us")
        ax.set_ylabel("CDF")
        ax.set_title("Decode Latency CDF")
        ax.legend(loc="lower right")
        _save_dual(fig, out_dir, "decode_latency_cdf")


def _timeline(trace_a: TraceData, trace_b: TraceData, out_dir: Path) -> None:
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)

        for trace, color in ((trace_a, WONG_PALETTE[5]), (trace_b, WONG_PALETTE[1])):
            idx = trace.df["row_idx"]
            rolling = trace.df["decode_dur_us"].rolling(100, min_periods=1).median()
            axes[0].plot(idx, rolling, color=color, label=f"{trace.name} rolling p50")

            miss_df = trace.df.loc[trace.df["deadline_met"] == 0]
            axes[0].scatter(
                miss_df["row_idx"],
                miss_df["decode_dur_us"],
                c=WONG_PALETTE[9],
                s=9,
                marker="x",
                alpha=0.6,
                label=f"{trace.name} misses",
            )

            axes[1].plot(
                idx,
                trace.df["sm_count"],
                color=color,
                linewidth=1.0,
                label=f"{trace.name} sm_count",
            )
            axes[1].plot(
                idx,
                trace.df["target_sm"],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=f"{trace.name} target_sm",
            )

        axes[0].set_ylabel("decode_dur_us")
        axes[0].set_title("Rolling Median Decode Latency (window=100)")
        axes[0].legend(ncol=2, loc="upper right")

        axes[1].set_ylabel("SM count")
        axes[1].set_xlabel("frame.slot index")
        axes[1].set_title("SM Allocation Timeline")
        axes[1].legend(ncol=2, loc="upper right")
        _save_dual(fig, out_dir, "timeline_decode_and_sm")


def _workload_vs_latency(
    trace_a: TraceData,
    trace_b: TraceData,
    out_dir: Path,
) -> None:
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)

        for ax, trace in zip(axes, (trace_a, trace_b)):
            met = trace.df["deadline_met"] == 1
            missed = ~met
            ax.scatter(
                trace.df.loc[met, "sum_tbs_bytes"],
                trace.df.loc[met, "decode_dur_us"],
                c=WONG_PALETTE[5],
                s=8,
                alpha=0.3,
                label="met",
            )
            ax.scatter(
                trace.df.loc[missed, "sum_tbs_bytes"],
                trace.df.loc[missed, "decode_dur_us"],
                c=WONG_PALETTE[9],
                s=8,
                alpha=0.3,
                label="missed",
            )
            ax.set_title(trace.name)
            ax.set_xlabel("sum_tbs_bytes")

        axes[0].set_ylabel("decode_dur_us")
        axes[1].legend(loc="upper left")
        fig.suptitle("Workload vs Decode Latency")
        _save_dual(fig, out_dir, "workload_vs_latency")


def _event_rate_series(length: int, rows, window: int = 100):
    flags = np.zeros(length, dtype=float)
    valid = rows[(rows >= 0) & (rows < length)]
    flags[valid.astype(int)] = 1.0
    return pd.Series(flags).rolling(window, min_periods=1).mean()


def _violation_inefficiency_timeline(
    trace_a: TraceData,
    trace_b: TraceData,
    out_dir: Path,
) -> None:
    with plt.rc_context(_paper_rc()):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True)

        for trace, base_color in ((trace_a, WONG_PALETTE[5]), (trace_b, WONG_PALETTE[1])):
            vio = trace.violations.copy()
            ine = trace.inefficiencies.copy()

            vio_delta = (vio["curr_sm"] - vio["prev_sm"]).to_numpy(dtype=float)
            ine_delta = -np.abs(
                (ine["curr_sm"] - ine["prev_sm"]).to_numpy(dtype=float)
            )

            axes[0].scatter(
                vio["row_idx"],
                vio_delta,
                marker="^",
                s=16,
                c=WONG_PALETTE[9],
                alpha=0.85,
                label=f"{trace.name} violations",
            )
            axes[0].scatter(
                ine["row_idx"],
                ine_delta,
                marker="v",
                s=16,
                c=WONG_PALETTE[5],
                alpha=0.85,
                label=f"{trace.name} inefficiencies",
            )

            length = len(trace.df)
            vio_rate = _event_rate_series(
                length,
                vio["row_idx"].to_numpy(dtype=int),
            )
            ine_rate = _event_rate_series(
                length,
                ine["row_idx"].to_numpy(dtype=int),
            )
            x = np.arange(length, dtype=int)
            axes[1].plot(
                x,
                vio_rate,
                color=base_color,
                linestyle="-",
                label=f"{trace.name} violation rate",
            )
            axes[1].plot(
                x,
                ine_rate,
                color=base_color,
                linestyle="--",
                label=f"{trace.name} inefficiency rate",
            )

        axes[0].axhline(0.0, color=WONG_PALETTE[0], linewidth=0.8)
        axes[0].set_ylabel("SM delta")
        axes[0].set_title("Violation / Inefficiency Events")
        axes[0].legend(ncol=2, loc="upper right")

        axes[1].set_ylabel("Rolling event rate")
        axes[1].set_xlabel("row index")
        axes[1].set_title("Rolling Event Rate (window=100)")
        axes[1].legend(ncol=2, loc="upper right")
        _save_dual(fig, out_dir, "violation_inefficiency_events")


def _summary(trace: TraceData) -> dict[str, float]:
    dur = trace.df["decode_dur_us"].to_numpy(dtype=float)
    deadline_miss = float((trace.df["deadline_met"] == 0).mean() * 100.0)
    return {
        "mean": float(np.mean(dur)),
        "p50": float(np.percentile(dur, 50)),
        "p95": float(np.percentile(dur, 95)),
        "p99": float(np.percentile(dur, 99)),
        "max": float(np.max(dur)),
        "deadline_miss_pct": deadline_miss,
        "violations": float(len(trace.violations)),
        "inefficiencies": float(len(trace.inefficiencies)),
    }


def _improvement(with_ctrl: float, without_ctrl: float) -> float:
    if without_ctrl == 0.0:
        return 0.0
    return ((without_ctrl - with_ctrl) / without_ctrl) * 100.0


def _print_summary(trace_with: TraceData, trace_without: TraceData) -> None:
    stats_with = _summary(trace_with)
    stats_without = _summary(trace_without)

    rows = [
        ("mean decode_dur_us", "mean"),
        ("p50 decode_dur_us", "p50"),
        ("p95 decode_dur_us", "p95"),
        ("p99 decode_dur_us", "p99"),
        ("max decode_dur_us", "max"),
        ("deadline miss %", "deadline_miss_pct"),
        ("violation count", "violations"),
        ("inefficiency count", "inefficiencies"),
    ]

    print("\nTrace Summary Statistics")
    print("=" * 96)
    print(
        f"{'Metric':<26}"
        f"{'with_ctrl':>18}"
        f"{'without_ctrl':>18}"
        f"{'improvement_%':>18}"
    )
    print("-" * 96)

    for label, key in rows:
        with_val = stats_with[key]
        without_val = stats_without[key]
        delta = _improvement(with_val, without_val)
        print(
            f"{label:<26}"
            f"{with_val:>18.3f}"
            f"{without_val:>18.3f}"
            f"{delta:>18.2f}"
        )
    print("=" * 96)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 5 publication-quality figures comparing LDPC real traces "
            "with/without control and print summary statistics."
        )
    )
    parser.add_argument(
        "--with-ctrl",
        type=Path,
        default=Path("data/ldpc_trace_with_ctrl.csv"),
        help="CSV trace path for run with control",
    )
    parser.add_argument(
        "--without-ctrl",
        type=Path,
        default=Path("data/ldpc_trace_without_ctrl.csv"),
        help="CSV trace path for run without control",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figure files",
    )
    parser.add_argument(
        "--total-sms",
        type=int,
        default=48,
        help="Total GPU SM count for adapter normalization",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with_ctrl = _load_trace(args.with_ctrl, args.total_sms, "with_ctrl")
    without_ctrl = _load_trace(args.without_ctrl, args.total_sms, "without_ctrl")

    _deadline_compliance(with_ctrl, without_ctrl, args.output_dir)
    _latency_cdf(with_ctrl, without_ctrl, args.output_dir)
    _timeline(with_ctrl, without_ctrl, args.output_dir)
    _workload_vs_latency(with_ctrl, without_ctrl, args.output_dir)
    _violation_inefficiency_timeline(with_ctrl, without_ctrl, args.output_dir)

    _print_summary(with_ctrl, without_ctrl)


if __name__ == "__main__":
    main()
