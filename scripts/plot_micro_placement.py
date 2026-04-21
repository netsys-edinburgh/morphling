#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false, reportMissingTypeArgument=false, reportUnannotatedClassAttribute=false, reportRedeclaration=false, reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table for micro placement overhead"
    )
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--output-tex", type=Path, default=None)
    parser.add_argument("--device-count", type=int, default=1024)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _fmt_latency_ms(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1.0:
        return "<1 ms"
    if value >= 60000.0:
        return f"{value / 60000.0:.1f} min"
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.1f} ms"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 0.01:
        return "<0.01\\%"
    return f"{value:.2f}\\%"


def _mode(row: dict[str, Any], key: str) -> dict[str, Any]:
    modes = row.get("modes", {})
    mode = modes.get(key, {}) if isinstance(modes, dict) else {}
    return mode if isinstance(mode, dict) else {}


def main() -> int:
    args = _parse_args()
    root = _repo_root()
    input_json = (
        args.input_json
        if args.input_json is not None
        else root / "results" / "microbenchmarks" / "placement_overhead.json"
    )
    output_tex = (
        args.output_tex
        if args.output_tex is not None
        else root / "figures" / "evaluation" / "table_micro_placement.tex"
    )
    output_tex.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    rows = payload.get("device_counts", {})
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"Invalid device_counts in {input_json}")

    key = str(args.device_count)
    if key not in rows:
        key = sorted(rows.keys())[-1]
    row = rows[key]
    if not isinstance(row, dict):
        raise ValueError(f"Bad row type for {key}")

    failed = row.get(
        "failed_count",
        payload.get("table_view", {}).get("churn_failures", 3),
    )

    cold = _mode(row, "cold_start")
    reuse = _mode(row, "reuse")
    incr = _mode(row, "incremental")
    greedy = _mode(row, "greedy")

    table = rf"""\begin{{table}}[t]
\centering
\small
\begin{{tabular}}{{lrrrr}}
\toprule
& \multicolumn{{2}}{{c}}{{Stable}} & \multicolumn{{2}}{{c}}{{Churn ({int(failed)} failures)}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}}
Mode & Latency & Exposed (\%) & Latency & Exposed (\%) \\
\midrule
Cold-start & {_fmt_latency_ms(cold.get("latency_ms"))} & {_fmt_pct(cold.get("batch_fraction_pct"))} & -- & -- \\
Reuse & {_fmt_latency_ms(reuse.get("latency_ms"))} & {_fmt_pct(reuse.get("batch_fraction_pct"))} & -- & -- \\
Incremental & -- & -- & {_fmt_latency_ms(incr.get("latency_ms"))} & {_fmt_pct(incr.get("batch_fraction_pct"))} \\
Greedy & -- & -- & {_fmt_latency_ms(greedy.get("latency_ms"))} & {_fmt_pct(greedy.get("batch_fraction_pct"))} \\
\bottomrule
\end{{tabular}}
\caption{{Placement-loop microbenchmark for \sys at {int(key)} devices (OPT-13B, batch size 128). Stable mode compares cold-start solving against assignment reuse; churn mode reports re-placement overhead after failures.}}
\label{{tab:micro-placement}}
\end{{table}}
"""

    output_tex.write_text(table + "\n", encoding="utf-8")
    print(f"[OK] wrote {output_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
