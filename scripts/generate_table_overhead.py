#!/usr/bin/env python3
"""Generate LaTeX table for overhead decomposition.

Usage:
  python3 scripts/generate_table_overhead.py \
      --data results/paper_data.json \
      --output figures/table_overhead.tex
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate overhead decomposition LaTeX table"
    )
    parser.add_argument(
        "--data", default="results/paper_data.json"
    )
    parser.add_argument(
        "--output-tex", default="figures/table_overhead.tex"
    )
    parser.add_argument(
        "--output-json", default="results/table_overhead.json"
    )
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    no = data.get("without_ctrl", {})
    we = data.get("with_ctrl", {})

    def get(d, key, default=0):
        return d.get(key, default)

    # Compute table rows
    rows = []

    # SM swaps per step
    no_swaps = get(no, "swap_count_mean")
    we_swaps = get(we, "swap_count_mean")
    reduction = (
        f"{(1 - we_swaps / no_swaps) * 100:.0f}\\%"
        if no_swaps > 0 else "---"
    )
    rows.append(("SM swaps per step", f"{no_swaps:.1f}",
                 f"{we_swaps:.1f}", reduction))

    # Avg overhead/swap (us)
    no_avg_us = get(no, "avg_swap_overhead_us")
    we_avg_us = get(we, "avg_swap_overhead_us")
    rows.append(("Avg.\\ overhead/swap ($\\mu$s)",
                 f"{no_avg_us:.1f}", f"{we_avg_us:.1f}", "---"))

    # Total swap overhead (ms)
    no_swap_ms = get(no, "swap_overhead_ms_per_step")
    we_swap_ms = get(we, "swap_overhead_ms_per_step")
    reduction = (
        f"{(1 - we_swap_ms / no_swap_ms) * 100:.0f}\\%"
        if no_swap_ms > 0 else "---"
    )
    rows.append(("Total swap overhead (ms)",
                 f"{no_swap_ms:.2f}", f"{we_swap_ms:.2f}", reduction))

    # Isolation violations/step
    no_viol = get(no, "violations_per_step_mean")
    we_viol = get(we, "violations_per_step_mean")
    reduction = (
        f"{(1 - we_viol / no_viol) * 100:.0f}\\%"
        if no_viol > 0 else "---"
    )
    rows.append(("Isolation violations/step",
                 f"{no_viol:.2f}", f"{we_viol:.2f}", reduction))

    # Avg violation dur (us)
    no_viol_time_ns = get(no, "total_violation_time_ns_mean")
    we_viol_time_ns = get(we, "total_violation_time_ns_mean")
    no_viol_count = max(no_viol, 1e-9)
    we_viol_count = max(we_viol, 1e-9)
    no_avg_viol_us = no_viol_time_ns / no_viol_count / 1000
    we_avg_viol_us = we_viol_time_ns / we_viol_count / 1000
    rows.append(("Avg.\\ violation dur.\\ ($\\mu$s)",
                 f"{no_avg_viol_us:.1f}", f"{we_avg_viol_us:.1f}", "---"))

    # Total violation time (ms)
    no_viol_ms = get(no, "violation_time_ms_per_step")
    we_viol_ms = get(we, "violation_time_ms_per_step")
    reduction = (
        f"{(1 - we_viol_ms / no_viol_ms) * 100:.0f}\\%"
        if no_viol_ms > 0 else "---"
    )
    rows.append(("Total violation time (ms)",
                 f"{no_viol_ms:.2f}", f"{we_viol_ms:.2f}", reduction))

    # Combined overhead
    no_combined = get(no, "combined_overhead_ms")
    we_combined = get(we, "combined_overhead_ms")
    reduction = (
        f"{(1 - we_combined / no_combined) * 100:.0f}\\%"
        if no_combined > 0 else "---"
    )
    rows.append(("\\textbf{Combined overhead (ms)}",
                 f"\\textbf{{{no_combined:.2f}}}",
                 f"\\textbf{{{we_combined:.2f}}}",
                 f"\\textbf{{{reduction}}}"))

    # Generate LaTeX
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{No Control} "
        "& \\textbf{\\sys} & \\textbf{Reduction} \\\\",
        "\\midrule",
    ]
    for name, nc, wv, red in rows:
        lines.append(f"{name} & {nc} & {wv} & {red} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    tex = "\n".join(lines)

    os.makedirs(os.path.dirname(args.output_tex) or ".", exist_ok=True)
    with open(args.output_tex, "w") as f:
        f.write(tex)
    print(f"Saved {args.output_tex}")

    # Machine-readable JSON
    table_data = {
        "rows": [
            {"metric": name, "no_ctrl": nc, "weaver": wv,
             "reduction": red}
            for name, nc, wv, red in rows
        ],
        "consistency_check": {
            "no_ctrl_combined": no_combined,
            "no_ctrl_swap_plus_viol": no_swap_ms + no_viol_ms,
            "within_5pct": abs(
                no_combined - (no_swap_ms + no_viol_ms)
            ) < 0.05 * max(no_combined, 0.001),
        },
    }
    os.makedirs(
        os.path.dirname(args.output_json) or ".", exist_ok=True
    )
    with open(args.output_json, "w") as f:
        json.dump(table_data, f, indent=2)
    print(f"Saved {args.output_json}")
    print(f"\nTable preview:\n{tex}")


if __name__ == "__main__":
    main()
