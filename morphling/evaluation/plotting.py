from __future__ import annotations

from pathlib import Path
from typing import Any

WARMUP_STEPS = 5
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


def _save_dual_output(fig: Any, plt: Any, stem_path: Path) -> None:
    fig.savefig(stem_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem_path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def _print_summary(results: dict[str, Any]) -> None:
    from .artifacts import _timed_rows

    print("\nBenchmark Summary (warmup skipped: first 5 steps)")
    print("=" * 78)
    print(f"{'mode':<12}{'mean_iter_ms':>22}{'mean_throughput':>22}{'final_loss':>22}")
    print("-" * 78)

    for mode in ("baseline", "greenctx"):
        if mode not in results:
            continue
        df = results[mode]
        timed = _timed_rows(df)
        mean_iter_ms = float(timed["wall_time_ms"].mean())
        mean_tput = float(timed["tokens_per_sec"].mean())
        final_loss = float(df["loss"].iloc[-1])
        print(f"{mode:<12}{mean_iter_ms:>22.3f}{mean_tput:>22.2f}{final_loss:>22.6f}")
    print("=" * 78)


def _plot_comparison(
    *,
    results: dict[str, Any],
    output_dir: Path,
    plt: Any,
    np: Any,
) -> None:
    from .artifacts import _timed_rows

    color_map = {"baseline": WONG_PALETTE[5], "greenctx": WONG_PALETTE[1]}
    marker_map = {"baseline": "o", "greenctx": "s"}

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = _timed_rows(results[mode])
            ax.plot(
                df["step_idx"],
                df["tokens_per_sec"],
                color=color_map[mode],
                marker=marker_map[mode],
                markersize=3,
                linewidth=1.2,
                label=mode,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Training Throughput")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_throughput")

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = results[mode]
            ax.plot(
                df["step_idx"],
                df["loss"],
                color=color_map[mode],
                marker=marker_map[mode],
                markersize=3,
                linewidth=1.2,
                label=mode,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_loss")

    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(3.33, 2.5))
        for mode in ("baseline", "greenctx"):
            if mode not in results:
                continue
            df = _timed_rows(results[mode])
            vals = df["wall_time_ms"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            bins = max(5, min(20, int(np.sqrt(len(vals)) * 2)))
            ax.hist(
                vals,
                bins=bins,
                alpha=0.60,
                color=color_map[mode],
                edgecolor="black",
                linewidth=0.4,
                label=mode,
            )
        ax.set_xlabel("Step wall time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Iteration Time Distribution")
        ax.legend()
        _save_dual_output(fig, plt, output_dir / "eval_iter_time")
