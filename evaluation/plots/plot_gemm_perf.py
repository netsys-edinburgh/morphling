# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportAny=false
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 3})
matplotlib.rcParams["axes.linewidth"] = 2
matplotlib.rcParams["axes.edgecolor"] = "black"

this_dir = os.path.join(os.path.dirname(__file__))
csv_path = os.path.join(this_dir, "..", "gemm_results.csv")
out_dir = os.path.join(this_dir, "figures")
out_path = os.path.join(out_dir, "gemm_perf.pdf")

df = pd.read_csv(csv_path)
perf_df = df[df["dim"].notna()].copy()
perf_df["dim"] = perf_df["dim"].astype(int)

dims = sorted(perf_df["dim"].unique().tolist())
if len(dims) < 2:
    raise ValueError("Expected at least 2 unique dims in gemm_results.csv")

partitions = sorted(perf_df["num_partitions"].unique().tolist())

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
markers = ["o", "s", "^", "D", "X"]

for idx, dim in enumerate(dims):
    dim_df = perf_df[perf_df["dim"] == dim].sort_values("num_partitions")
    x = dim_df["num_partitions"].to_numpy()
    y = dim_df["GFLOPS"].to_numpy()

    axes[0].plot(
        x,
        y,
        marker=markers[idx % len(markers)],
        color=colors[idx % len(colors)],
        label=f"dim={dim}",
    )

    base = dim_df.loc[dim_df["num_partitions"] == 1, "GFLOPS"]
    if not base.empty and base.iloc[0] != 0:
        efficiency = y / base.iloc[0]
        axes[1].plot(
            x,
            efficiency,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
        )

axes[0].set_ylabel("GFLOPS")
axes[0].grid(True)
axes[0].legend(loc="best")

axes[1].set_xlabel("Number of SM partitions")
axes[1].set_ylabel("Scaling efficiency")
axes[1].set_xticks(partitions)
axes[1].grid(True)

plt.tight_layout()
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
plt.close(fig)
