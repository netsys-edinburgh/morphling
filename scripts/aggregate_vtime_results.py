#!/usr/bin/env python3
"""Aggregate per-baseline VTIME runtime JSONs into a single summary.

Reads the output of run_baseline_comparison.py (with --vtime-log)
for each model × baseline, and produces a summary JSON suitable
for the e2e latency plot.

Usage:
  python3 scripts/aggregate_vtime_results.py \
    --vtime-root results/vtime_models \
    --output results/vtime_models/vtime_summary.json

Also supports reading pre-existing OPT-125M VTIME data from
results/vtime/summary/paper_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BASELINES = ["cleave", "dtfm", "asteroid", "confident", "alpa"]

# Model display names for the plot
MODEL_LABELS = {
    "opt-125m": "OPT-125M",
    "opt-1.3b": "OPT-1.3B",
    "opt-13b": "OPT-13B",
    "opt-30b": "OPT-30B",
    "opt-66b": "OPT-66B",
    "llama2-7b": "Llama2-7B",
    "llama2-13b": "Llama2-13B",
    "llama2-70b": "Llama2-70B",
}

# Ordered for the plot (ascending model size)
MODEL_ORDER = [
    "opt-125m",
    "opt-1.3b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama2-7b",
    "llama2-13b",
    "llama2-70b",
]


def _find_runtime_json(
    vtime_root: Path,
    model: str,
    baseline: str,
) -> Path | None:
    """Locate the runtime JSON produced by post-processing."""
    results_dir = vtime_root / model / f"{baseline}_results"
    candidate = results_dir / "runtimes" / f"{baseline}_runtime.json"
    if candidate.exists():
        return candidate
    return None


def _extract_total_ms(runtime_path: Path) -> float | None:
    """Pull total_runtime_ms from a *_runtime.json file."""
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("total_runtime_ms")


def _load_opt125m_vtime(
    paper_summary: Path,
) -> dict[str, float]:
    """Load pre-existing OPT-125M VTIME from paper_summary.json."""
    if not paper_summary.exists():
        return {}
    data = json.loads(paper_summary.read_text(encoding="utf-8"))
    results = data.get("results", {})
    out: dict[str, float] = {}
    for baseline in BASELINES:
        entry = results.get(baseline, {})
        total = entry.get("total_ms")
        if total is not None:
            out[baseline] = float(total)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate VTIME runtime results"
    )
    parser.add_argument(
        "--vtime-root",
        required=True,
        type=str,
        help="Root dir with model/baseline_results/",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output summary JSON path",
    )
    parser.add_argument(
        "--include-opt125m",
        action="store_true",
        default=True,
        help="Include OPT-125M from paper_summary.json",
    )
    args = parser.parse_args()

    vtime_root = Path(args.vtime_root)
    summary: dict[str, dict[str, dict[str, float | None]]] = {}

    for model in MODEL_ORDER:
        model_data: dict[str, dict[str, float | None]] = {}
        for baseline in BASELINES:
            rt_path = _find_runtime_json(vtime_root, model, baseline)
            if rt_path is None:
                print(f"[MISS] {model}/{baseline}: no runtime JSON")
                model_data[baseline] = {"total_runtime_ms": None}
                continue

            total = _extract_total_ms(rt_path)
            if total is None:
                print(
                    f"[WARN] {model}/{baseline}: "
                    f"no total_runtime_ms in {rt_path}"
                )
            else:
                print(f"[OK]   {model}/{baseline}: {total:.1f} ms")
            model_data[baseline] = {"total_runtime_ms": total}

        summary[model] = {"baselines": model_data}

    if args.include_opt125m and "opt-125m" in summary:
        has_data = any(
            v.get("total_runtime_ms") is not None
            for v in summary["opt-125m"]["baselines"].values()
        )
        if not has_data:
            paper_path = Path("results/vtime/summary/paper_summary.json")
            opt125m = _load_opt125m_vtime(paper_path)
            if opt125m:
                summary["opt-125m"] = {
                    "baselines": {
                        b: {"total_runtime_ms": opt125m.get(b)}
                        for b in BASELINES
                    },
                }
                print(f"[FALL] opt-125m: no 64-dev data, using {paper_path}")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "description": ("VTIME batch runtime comparison across models"),
        "mode": "vtime",
        "num_devices": 64,
        "model_order": [MODEL_LABELS.get(m, m) for m in MODEL_ORDER],
        "baseline_order": BASELINES,
        "models": {},
    }

    for model in MODEL_ORDER:
        if model not in summary:
            continue
        label = MODEL_LABELS.get(model, model)
        output["models"][label] = summary[model]

    out_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
