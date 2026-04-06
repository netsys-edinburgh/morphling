from __future__ import annotations

import csv
import json
from pathlib import Path

from .training_sim import SimulationResult


def format_comparison_table(results: dict[str, SimulationResult]) -> str:
    headers = [
        "Topology",
        "Mean Step (ms)",
        "Std Step (ms)",
        "Throughput (steps/s)",
        "Comm Overhead (%)",
        "Slowest Device",
    ]

    rows: list[list[str]] = []
    for topology, result in results.items():
        slowest = result.slowest_device_rank()
        rows.append(
            [
                topology,
                f"{result.mean_step_time_s * 1e3:.3f}",
                f"{result.std_step_time_s * 1e3:.3f}",
                f"{result.throughput_steps_per_s:.4f}",
                f"{result.communication_overhead_pct():.2f}",
                "-" if slowest is None else str(slowest),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt(headers), sep]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def export_json(results: dict[str, SimulationResult], path: str) -> None:
    output: dict[str, dict[str, object]] = {}
    for topology, result in results.items():
        steps_payload: list[dict[str, object]] = []
        for step in result.steps:
            per_device = {
                rank: {
                    "fwd_time_s": step.fwd_time_s[rank],
                    "bwd_time_s": step.bwd_time_s[rank],
                    "grad_sync_time_s": step.grad_sync_time_s[rank],
                    "total_time_s": step.total_time_s[rank],
                }
                for rank in sorted(step.total_time_s)
            }
            steps_payload.append(
                {
                    "step_idx": step.step_idx,
                    "step_time_s": step.step_time_s,
                    "devices": per_device,
                }
            )

        output[topology] = {
            "topology": result.topology,
            "mean_step_time_s": result.mean_step_time_s,
            "std_step_time_s": result.std_step_time_s,
            "throughput_steps_per_s": result.throughput_steps_per_s,
            "communication_overhead_pct": result.communication_overhead_pct(),
            "slowest_device_rank": result.slowest_device_rank(),
            "steps": steps_payload,
        }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def export_csv(results: dict[str, SimulationResult], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "topology",
                "step_idx",
                "step_time_s",
                "device_rank",
                "fwd_time_s",
                "bwd_time_s",
                "grad_sync_time_s",
                "total_time_s",
            ],
        )
        writer.writeheader()
        for topology, result in results.items():
            for step in result.steps:
                for rank in sorted(step.total_time_s):
                    writer.writerow(
                        {
                            "topology": topology,
                            "step_idx": step.step_idx,
                            "step_time_s": step.step_time_s,
                            "device_rank": rank,
                            "fwd_time_s": step.fwd_time_s[rank],
                            "bwd_time_s": step.bwd_time_s[rank],
                            "grad_sync_time_s": step.grad_sync_time_s[rank],
                            "total_time_s": step.total_time_s[rank],
                        }
                    )
