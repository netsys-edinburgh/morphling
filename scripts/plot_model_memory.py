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
"""Per-device memory analytical model (fig:eval-e2e-memory).

Computes peak per-device training memory for five distributed training
systems across model sizes from OPT-125M to OPT-66B.

Memory models:
  - Wasp (Cleave): GEMM tile sharding.  Devices hold only the working
    set for the current tile (input sub-matrix, weight sub-matrix,
    output sub-matrix).  No optimizer or gradient state on device.
  - DTFM:      PP=2 fixed, layer-granularity.
  - Asteroid:  PP=8 fixed, layer-granularity (HPP planner).
  - Confidant: PP=best_pp(N, L), maximises pipeline stages.
  - Alpa:      PP=4 (latency-optimal auto-tuning; low PP preferred
               because inter-stage comm dominates on mobile links).

All layer-parallel systems carry W + G + 2W_opt + K_p * A per device
(formula from baselines.core.memory.estimate_training_memory_mb).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, cast

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

# ── Model architectures ──────────────────────────────────────────────

MODELS: list[tuple[str, dict[str, int]]] = [
    ("OPT-125M", {"L": 12, "d": 768, "d_ff": 3072}),
    ("OPT-1.3B", {"L": 24, "d": 2048, "d_ff": 8192}),
    ("Llama2-7B", {"L": 32, "d": 4096, "d_ff": 11008}),
    ("OPT-13B", {"L": 40, "d": 5120, "d_ff": 20480}),
    ("OPT-30B", {"L": 48, "d": 7168, "d_ff": 28672}),
    ("OPT-66B", {"L": 64, "d": 9216, "d_ff": 36864}),
]

# ── System config ────────────────────────────────────────────────────

SYSTEM_ORDER = ["dtfm", "alpa", "asteroid", "confident", "cleave"]

SYSTEM_LABELS: dict[str, str] = {
    "cleave": "Wasp",
    "dtfm": "DTFM",
    "asteroid": "Asteroid",
    "confident": "Confidant",
    "alpa": "Alpa",
}

SYSTEM_COLORS: dict[str, str] = {
    "dtfm": "#f1f2fa",
    "asteroid": "#a3aae3",
    "confident": "#1994dc",
    "alpa": "#463fc4",
    "cleave": "#000000",
}

# ── Global parameters ────────────────────────────────────────────────

N_AVAILABLE = 8192
BATCH_SIZE = 16
SEQ_LEN = 512
MICRO_BATCH = 1
BYTES_PER_PARAM = 2  # fp16
SAFETY_FACTOR = 1.10
MOBILE_BUDGET_GB = 0.5


# ── Helper: best (PP, DP) factorisation ──────────────────────────────


def _best_pp_dp(
    num_devices: int,
    num_layers: int,
) -> tuple[int, int]:
    """Largest PP <= num_layers that divides num_devices."""
    best_pp, best_dp = 1, num_devices
    for pp in range(min(num_devices, num_layers), 0, -1):
        if num_devices % pp == 0:
            best_pp = pp
            best_dp = num_devices // pp
            break
    return best_pp, best_dp


# ── Memory model: layer-parallel systems ─────────────────────────────


def _layer_parallel_memory_gb(
    num_layers: int,
    d: int,
    d_ff: int,
    pp: int,
    tp: int = 1,
    stage_idx: int = 0,
) -> float:
    """Peak per-device training memory (GB).

    Formula (matches ``baselines.core.memory``)::

        Mem = (W + G + 2W_opt + K_p * A) * safety

    Stage 0 is the worst case because K_p = 2*(PP - 0) - 1 is maximal.
    """
    layers_in_stage = max(1, (num_layers + pp - 1) // pp)

    params_per_layer = 4 * d * d + 2 * d * d_ff
    stage_params = params_per_layer * layers_in_stage / tp

    mem_W = stage_params * BYTES_PER_PARAM
    mem_G = mem_W
    mem_OPT = 2.0 * mem_W  # AdamW first + second moment

    act_per_sample_layer = SEQ_LEN * d * BYTES_PER_PARAM
    act_per_device = act_per_sample_layer * layers_in_stage / tp * MICRO_BATCH

    k_p = max(1, 2 * (pp - stage_idx) - 1)

    total_bytes = mem_W + mem_G + mem_OPT + k_p * act_per_device
    return total_bytes / (1024.0**3) * SAFETY_FACTOR


# ── Memory model: Cleave / Wasp ─────────────────────────────────────


def _cleave_memory_gb(d: int, d_ff: int) -> float:
    """Per-device memory (GB) under GEMM tile sharding.

    Devices receive input and weight sub-matrices for a single GEMM
    tile, compute the output sub-tile, and return it.  No optimizer,
    gradient, or activation-buffer state lives on device.

    Peak memory = largest tile across forward-pass GEMMs (FFN up/down
    projections dominate).
    """
    M = BATCH_SIZE * SEQ_LEN

    peak_bytes = 0.0
    for n_common, q_out in [(d, d_ff), (d_ff, d)]:
        total_area = float(M * q_out)
        area = total_area / N_AVAILABLE
        if area <= 0.0:
            continue

        alpha = math.sqrt(area * M / q_out) if q_out > 0 else 0.0
        beta = area / max(alpha, 1e-12)

        if alpha > M:
            alpha = float(M)
            beta = area / max(alpha, 1e-12)
        if beta > q_out:
            beta = float(q_out)
            alpha = area / max(beta, 1e-12)

        tile_bytes = BYTES_PER_PARAM * (
            alpha * n_common + n_common * beta + alpha * beta
        )
        peak_bytes = max(peak_bytes, tile_bytes)

    runtime_overhead = 5.0 * 1024 * 1024  # 5 MB CUDA/framework overhead
    return (peak_bytes + runtime_overhead) / (1024.0**3)


# ── Top-level dispatcher ────────────────────────────────────────────

# PP configuration per system.  These match the strategy defaults used
# in project_analytical_scaling.py and run_baseline_comparison.py.
_FIXED_PP: dict[str, int] = {
    "dtfm": 2,  # DTFM paper: 2-stage pipeline
    "asteroid": 8,  # HPP planner default
    "alpa": 4,  # latency-optimal ILP tends to low PP on mobile
}


def compute_memory_gb(
    system: str,
    num_layers: int,
    d: int,
    d_ff: int,
) -> float:
    """Peak per-device memory (GB) for *system* on a given model."""
    if system == "cleave":
        return _cleave_memory_gb(d, d_ff)

    if system == "confident":
        pp = _best_pp_dp(N_AVAILABLE, num_layers)[0]
    else:
        pp = _FIXED_PP[system]

    return _layer_parallel_memory_gb(num_layers, d, d_ff, pp=pp)


# ── Plotting ─────────────────────────────────────────────────────────


def _bump_font_sizes(delta: float) -> None:
    keys = [
        "font.size",
        "axes.labelsize",
        "axes.titlesize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ]
    for key in keys:
        val = plt.rcParams.get(key)
        if isinstance(val, (int, float)):
            plt.rcParams[key] = float(val) + delta


def plot_model_memory() -> None:
    model_labels = [m[0] for m in MODELS]
    n_models = len(model_labels)
    n_systems = len(SYSTEM_ORDER)

    data: dict[str, list[float]] = {s: [] for s in SYSTEM_ORDER}
    for _, cfg in MODELS:
        for system in SYSTEM_ORDER:
            mem = compute_memory_gb(system, cfg["L"], cfg["d"], cfg["d_ff"])
            data[system].append(mem)

    header = f"{'Model':<12}"
    for s in SYSTEM_ORDER:
        header += f"  {SYSTEM_LABELS[s]:>10}"
    print(header)
    for i, (label, _) in enumerate(MODELS):
        row = f"{label:<12}"
        for s in SYSTEM_ORDER:
            row += f"  {data[s][i]:>10.4f}"
        print(row)

    out_path = Path("figures/evaluation/model_memory.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with paper_style(width=7.0, height=2.8):
        _bump_font_sizes(2.0)
        fig = cast(Any, plt.figure())
        ax = fig.add_subplot(111)

        x = np.arange(n_models)
        width = min(0.15, 0.8 / n_systems)
        offsets = (np.arange(n_systems) - (n_systems - 1) / 2.0) * width

        legend_handles: list[Patch] = []
        for idx, system in enumerate(SYSTEM_ORDER):
            values = np.array(data[system], dtype=float)
            color = SYSTEM_COLORS[system]
            ax.bar(
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
                    label=SYSTEM_LABELS[system],
                )
            )

        ax.axhline(
            MOBILE_BUDGET_GB,
            color="red",
            linestyle="--",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            "0.5 GB budget",
            xy=(n_models - 0.5, MOBILE_BUDGET_GB),
            xytext=(0, 6),
            textcoords="offset points",
            color="red",
            fontsize=8,
            ha="right",
            va="bottom",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("Per-device memory (GB)")
        ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.85),
            ncol=n_systems,
            frameon=False,
        )
        fig.subplots_adjust(top=0.80)
        save_dual_output(fig, out_path, None)
        plt.close(fig)


if __name__ == "__main__":
    plot_model_memory()
