# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("classic")
matplotlib.rcParams.update({"font.size": 28, "lines.linewidth": 1})

this_dir = os.path.join(os.path.dirname(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
data_dir = os.path.join(repo_root, "data")

random_df = pd.read_csv(os.path.join(data_dir, "trace_random.csv"))
shaped_df = pd.read_csv(os.path.join(data_dir, "trace_shaped.csv"))

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)  # pyright: ignore[reportAny]

axes[0].plot(random_df.index, random_df["sm_count"])  # pyright: ignore[reportAny]
axes[0].set_title("Random Traffic")  # pyright: ignore[reportAny]
axes[0].set_ylabel("SM Count")  # pyright: ignore[reportAny]
axes[0].grid()  # pyright: ignore[reportAny]

axes[1].plot(shaped_df.index, shaped_df["sm_count"])  # pyright: ignore[reportAny]
axes[1].set_title("Shaped Traffic")  # pyright: ignore[reportAny]
axes[1].set_xlabel("Slot Index")  # pyright: ignore[reportAny]
axes[1].set_ylabel("SM Count")  # pyright: ignore[reportAny]
axes[1].grid()  # pyright: ignore[reportAny]

plt.tight_layout()

figure_dir = os.path.join(this_dir, "figures")
os.makedirs(figure_dir, exist_ok=True)
plt.savefig(
    os.path.join(figure_dir, "trace_timeseries.pdf"), bbox_inches="tight"
)
plt.close()
