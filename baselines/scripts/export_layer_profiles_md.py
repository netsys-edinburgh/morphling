#!/usr/bin/env python3
"""Export all node layer_profile entries into a Markdown report."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path
from typing import Any


def _parse_rank(path: str, payload: dict[str, Any]) -> int:
    rank = payload.get("rank")
    if isinstance(rank, int):
        return rank
    stem = Path(path).stem
    if "_rank" in stem:
        try:
            return int(stem.split("_rank")[-1])
        except ValueError:
            pass
    return 10**9


def _fmt_num(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _fmt_ts(value: Any) -> str:
    try:
        ts = float(value)
        return dt.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return "-"


def _append_layer_profile_table(
    lines: list[str], layer_profile: dict[str, Any],
) -> None:
    if not isinstance(layer_profile, dict) or not layer_profile:
        lines.append("_No layer_profile data found._")
        lines.append("")
        return

    batch_keys = sorted(
        layer_profile.keys(),
        key=lambda k: int(k) if str(k).isdigit() else str(k),
    )

    lines.append(
        "| batch_size | latency_ms_mean | latency_ms_p50 | latency_ms_p90 | "
        "latency_ms_std | tokens_per_sec | estimated_total_latency_ms | "
        "estimated_model_tokens_per_sec | num_layers |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for bs in batch_keys:
        metrics = layer_profile.get(bs, {})
        if not isinstance(metrics, dict):
            continue
        lines.append(
            f"| {bs} | "
            f"{_fmt_num(metrics.get('latency_ms_mean'))} | "
            f"{_fmt_num(metrics.get('latency_ms_p50'))} | "
            f"{_fmt_num(metrics.get('latency_ms_p90'))} | "
            f"{_fmt_num(metrics.get('latency_ms_std'))} | "
            f"{_fmt_num(metrics.get('tokens_per_sec'))} | "
            f"{_fmt_num(metrics.get('estimated_total_latency_ms'))} | "
            f"{_fmt_num(metrics.get('estimated_model_tokens_per_sec'))} | "
            f"{_fmt_num(metrics.get('num_layers'))} |"
        )
    lines.append("")


def generate_markdown(profiles_dir: Path, output_path: Path) -> int:
    files = sorted(glob.glob(str(profiles_dir / "profile_*.json")))
    if not files:
        raise FileNotFoundError(
            f"No profile JSON files found in: {profiles_dir}"
        )

    records: list[tuple[int, str, dict[str, Any]]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records.append((_parse_rank(path, payload), path, payload))

    records.sort(key=lambda x: x[0])

    lines: list[str] = []
    lines.append("# Layer Profile Report")
    lines.append("")
    lines.append(f"- Generated: `{dt.datetime.now().isoformat(sep=' ', timespec='seconds')}`")
    lines.append(f"- Profiles dir: `{profiles_dir}`")
    lines.append(f"- Nodes: `{len(records)}`")
    lines.append("")

    for rank, path, payload in records:
        host = payload.get("hostname", "-")
        ts = _fmt_ts(payload.get("timestamp_unix"))
        net_ts = _fmt_ts(payload.get("network_timestamp_unix"))
        model = payload.get("model", {})
        layer_profile = payload.get("layer_profile", {})
        layer_profiles_by_mps = payload.get(
            "layer_profiles_by_mps", {}
        )

        lines.append(f"## Rank {rank} ({host})")
        lines.append("")
        lines.append(f"- Source: `{path}`")
        lines.append(f"- Profile timestamp: `{ts}`")
        lines.append(f"- Network timestamp: `{net_ts}`")
        lines.append(
            "- Model: "
            f"`layers={model.get('num_layers', '-')}, "
            f"embed_dim={model.get('embed_dim', '-')}, "
            f"heads={model.get('num_heads', '-')}, "
            f"d_ff={model.get('d_ff', '-')}, "
            f"seq_len={model.get('seq_len', '-')}`"
        )
        lines.append("")

        if isinstance(layer_profiles_by_mps, dict) and layer_profiles_by_mps:
            mps_keys = sorted(
                layer_profiles_by_mps.keys(),
                key=lambda k: int(k) if str(k).isdigit() else str(k),
            )
            lines.append(
                f"- MPS profiles: `{', '.join(str(k) for k in mps_keys)}`"
            )
            lines.append("")
            for mps_key in mps_keys:
                lines.append(f"### MPS {mps_key}%")
                lines.append("")
                lp = layer_profiles_by_mps.get(mps_key, {})
                if not isinstance(lp, dict):
                    lp = {}
                _append_layer_profile_table(lines, lp)
            continue

        _append_layer_profile_table(lines, layer_profile)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export profile_*.json layer_profile values to Markdown."
    )
    parser.add_argument(
        "--profiles-dir",
        default="baselines/profiles",
        help="Directory containing profile_*.json files.",
    )
    parser.add_argument(
        "--output",
        default="baselines/profiles/layer_profiles_report.md",
        help="Path to output markdown file.",
    )
    args = parser.parse_args()

    count = generate_markdown(
        profiles_dir=Path(args.profiles_dir),
        output_path=Path(args.output),
    )
    print(f"Saved markdown report for {count} node profiles: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
