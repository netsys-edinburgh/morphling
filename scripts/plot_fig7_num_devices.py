#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = Path(
    "/mnt/data/yufeng.xia/project/AgentSkillsArxiv/skills/conference-plot/scripts"
)
sys.path.insert(0, str(SKILL_SCRIPTS))

from plot_utils import paper_style, save_dual_output

DEFAULT_SOURCE_ROOT = REPO / "results" / "device_scaling" / "opt-13b"
DEFAULT_OUT_JSON = (
    REPO / "results" / "vtime_scaling_fig7" / "num_devices_device_scaling.json"
)
DEFAULT_OUT_PDF = (
    REPO / "figures" / "evaluation" / "fig7_num_devices_runtime.pdf"
)
DEFAULT_OUT_PNG = (
    REPO / "figures" / "evaluation" / "fig7_num_devices_runtime.png"
)

DEFAULT_DEVICE_COUNTS = [32, 64, 128, 256, 512, 1024]
METHODS = ["alpa", "dtfm", "cleave"]
METHOD_LABELS = {"alpa": "Alpa", "dtfm": "DTFM", "cleave": "Cleave"}
METHOD_COLORS = {
    "alpa": "#d9e7f5",
    "dtfm": "#7fb3ea",
    "cleave": "#1675e0",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 7 device scaling")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Directory containing n*/summary.json subdirectories",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Path to write extracted plotting data JSON",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=DEFAULT_OUT_PDF,
        help="Path to write PDF figure",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=DEFAULT_OUT_PNG,
        help="Path to write PNG figure",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default="32,64,128,256,512,1024",
        help="Comma-separated device counts",
    )
    return parser.parse_args()


def collect_points(source_root: Path, device_counts: list[int]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": str(source_root.relative_to(REPO)) + "/n*/summary.json",
        "model": "opt-13b",
        "batch_size": 128,
        "seq_len": 1024,
        "units": {
            "raw": "ms",
            "plot_y": "100s",
            "plot_y_definition": "total_runtime_ms / 100000.0",
        },
        "points": [],
    }

    for count in device_counts:
        summary = load_json(source_root / f"n{count}" / "summary.json")
        baselines = summary.get("baselines", {})
        row = {"num_devices": count, "baselines": {}}
        for method in METHODS:
            stats = baselines.get(method, {})
            total_ms = (
                float(stats["total_runtime_ms"])
                if isinstance(stats, dict)
                and isinstance(stats.get("total_runtime_ms"), (int, float))
                else None
            )
            row["baselines"][method] = {
                "total_runtime_ms": total_ms,
                "runtime_100s": (
                    None if total_ms is None else total_ms / 100000.0
                ),
            }
        payload["points"].append(row)

    return payload


def plot(
    payload: dict[str, Any],
    out_pdf: Path,
    out_png: Path,
    device_counts: list[int],
) -> None:
    with paper_style(width=7.0, height=2.5, font_size=9):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ]
        fig, ax = plt.subplots()
        x = np.arange(len(device_counts), dtype=float)
        width = 0.16
        offsets = {"alpa": -width, "dtfm": 0.0, "cleave": width}

        for method in METHODS:
            xs: list[float] = []
            ys: list[float] = []
            missing_xs: list[float] = []
            for i, row in enumerate(payload["points"]):
                value = row["baselines"][method]["runtime_100s"]
                xpos = x[i] + offsets[method]
                if value is None:
                    missing_xs.append(xpos)
                    continue
                xs.append(xpos)
                ys.append(float(value))

            if xs:
                ax.bar(
                    xs,
                    ys,
                    width=width,
                    color=METHOD_COLORS[method],
                    edgecolor="black",
                    linewidth=0.8,
                    label=METHOD_LABELS[method],
                )

            for xpos in missing_xs:
                ax.text(
                    xpos,
                    1.0,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#666666",
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in device_counts])
        ax.set_xlabel("Number of Devices")
        ax.set_ylabel("Runtime (100s)")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", linestyle=(0, (4, 2)), linewidth=0.8, color="black")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.72, 0.995),
            ncol=3,
            frameon=True,
        )

        save_dual_output(fig, out_pdf, out_png, save_both=False)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    device_counts = [int(x) for x in args.counts.split(",") if x.strip()]
    payload = collect_points(args.source_root, device_counts)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(payload, args.out_pdf, args.out_png, device_counts)
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_pdf}")
    print(f"Saved {args.out_png}")


if __name__ == "__main__":
    main()
