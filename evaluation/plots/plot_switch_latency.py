import os
from csv import DictReader

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 3})
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "black"

this_dir = os.path.join(os.path.dirname(__file__))
csv_path = os.path.join(this_dir, "..", "switch_results.csv")
out_dir = os.path.join(this_dir, "figures")
out_path = os.path.join(out_dir, "switch_latency.pdf")

rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = DictReader(f)
    for row in reader:
        benchmark_names = [
            b.strip() for b in row["benchmark_name"].split("+") if b.strip()
        ]
        if not benchmark_names:
            benchmark_names = [row["benchmark_name"]]

        for benchmark_name in benchmark_names:
            rows.append(
                {
                    "trace_type": row["trace_type"],
                    "benchmark_name": benchmark_name,
                    "switch_p50_ns": float(row["switch_p50_ns"]),
                    "switch_p99_ns": float(row["switch_p99_ns"]),
                    "switch_max_ns": float(row["switch_max_ns"]),
                    "violations": float(row["violations"]),
                    "violation_rate": float(row["violation_rate"]),
                    "actual_ctx_switches": float(row["actual_ctx_switches"]),
                }
            )

rows_by_key = {
    (r["trace_type"], r["benchmark_name"]): r for r in rows
}

trace_order = ["random", "shaped"]
benchmark_order = ["ReplayViolation", "ForcedSwitch"]

categories = []
for trace_type in trace_order:
    for benchmark_name in benchmark_order:
        key = (trace_type, benchmark_name)
        if key not in rows_by_key:
            continue
        match = rows_by_key[key]
        categories.append(
            {
                "trace_type": trace_type,
                "benchmark_name": benchmark_name,
                "switch_p50_ns": float(match["switch_p50_ns"]),
                "switch_p99_ns": float(match["switch_p99_ns"]),
                "switch_max_ns": float(match["switch_max_ns"]),
                "violations": float(match["violations"]),
                "violation_rate": float(match["violation_rate"]),
                "actual_ctx_switches": float(match["actual_ctx_switches"]),
            }
        )

labels = [f"{c['trace_type']}\n{c['benchmark_name']}" for c in categories]
x = np.arange(len(categories))

p50 = np.array([c["switch_p50_ns"] for c in categories])
p99 = np.array([c["switch_p99_ns"] for c in categories])
max_vals = np.array([c["switch_max_ns"] for c in categories])
max_err_p50 = np.maximum(max_vals - p50, 0)
max_err_p99 = np.maximum(max_vals - p99, 0)

ctx_switches = np.array([c["actual_ctx_switches"] for c in categories])
violations = np.array([c["violations"] for c in categories])
violation_rates = np.array([c["violation_rate"] for c in categories])

fig, (ax_latency, ax_meta) = plt.subplots(
    2,
    1,
    figsize=(9, 8),
    sharex=True,
    gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
)

width = 0.36
ax_latency.bar(
    x - width / 2,
    p50,
    width,
    label="p50",
    color="#4C72B0",
    yerr=max_err_p50,
    capsize=4,
)
ax_latency.bar(
    x + width / 2,
    p99,
    width,
    label="p99",
    color="#55A868",
    yerr=max_err_p99,
    capsize=4,
)
ax_latency.set_ylabel("Latency (ns)")
ax_latency.grid(axis="y", alpha=0.35)
ax_latency.legend(loc="upper left", fontsize=18)

for i, max_val in enumerate(max_vals):
    ax_latency.text(
        x[i],
        max_val * 1.02,
        f"max={int(max_val)}",
        ha="center",
        va="bottom",
        fontsize=12,
        rotation=90,
    )

ax_meta.bar(x, ctx_switches, width=0.56, color="#C44E52", alpha=0.8)
ax_meta.set_ylabel("ctx\nswitches")
ax_meta.set_xlabel("Trace Type × Benchmark")
ax_meta.grid(axis="y", alpha=0.35)

for i in range(len(categories)):
    ax_meta.text(
        x[i],
        ctx_switches[i],
        (
            f"viol={int(violations[i])}\n"
            f"rate={violation_rates[i]:.4f}"
        ),
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax_meta_rate = ax_meta.twinx()
ax_meta_rate.plot(x, violation_rates, color="black", marker="o", linewidth=2)
ax_meta_rate.set_ylabel("viol. rate")
ax_meta_rate.set_ylim(0, max(0.01, float(np.max(violation_rates)) * 1.3 + 1e-6))

ax_meta.set_xticks(x)
ax_meta.set_xticklabels(labels, rotation=15, ha="right")

os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)
