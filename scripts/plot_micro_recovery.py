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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot micro recovery breakdown figure"
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-pdf", type=Path, default=None)
    parser.add_argument("--width", type=float, default=3.5)
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
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


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
        else root / "results" / "microbenchmarks" / "recovery_breakdown.json"
    )
    output_pdf = (
        args.output_pdf
        if args.output_pdf is not None
        else root / "figures" / "evaluation" / "micro_recovery.pdf"
    )

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    scenarios = payload.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        raise ValueError(f"Invalid scenarios in {input_json}")

    keys = sorted(scenarios.keys(), key=lambda x: int(x))
    labels = [f"{int(k)} failed" for k in keys]

    detect: list[float] = []
    resolve: list[float] = []
    redispatch: list[float] = []
    recompute: list[float] = []
    for k in keys:
        phases = scenarios.get(k, {}).get("phases_ms", {})
        detect.append(float(phases.get("detection", 0.0)))
        resolve.append(float(phases.get("re_solve", 0.0)))
        redispatch.append(float(phases.get("re_dispatch", 0.0)))
        recompute.append(float(phases.get("recompute", 0.0)))

    y = np.arange(len(labels))
    with matplotlib.rc_context(_style(float(args.width), float(args.height))):
        fig, ax = plt.subplots()
        left = np.zeros(len(labels), dtype=float)

        colors = {
            "detection": "#999999",
            "re_solve": "#0072B2",
            "re_dispatch": "#D55E00",
            "recompute": "#009E73",
        }

        ax.barh(y, detect, left=left, color=colors["detection"], label="Detection")
        left += np.array(detect, dtype=float)
        ax.barh(y, resolve, left=left, color=colors["re_solve"], label="Re-solve")
        left += np.array(resolve, dtype=float)
        ax.barh(
            y,
            redispatch,
            left=left,
            color=colors["re_dispatch"],
            label="Re-dispatch",
        )
        left += np.array(redispatch, dtype=float)
        ax.barh(
            y,
            recompute,
            left=left,
            color=colors["recompute"],
            label="Recompute",
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Recovery time (ms)")
        ax.set_ylabel("Failure scenario")
        ax.grid(True, axis="x")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.28),
            ncol=2,
            frameon=False,
        )

        _save(fig, output_pdf)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
