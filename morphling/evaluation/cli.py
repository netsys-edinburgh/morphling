from __future__ import annotations

import argparse
import logging

from .runner import _run_benchmark


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU OPT training benchmark with green context",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        default="data/ldpc_trace_with_ctrl.csv",
        help="LDPC CSV trace path (raw format)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of optimizer steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/",
        help="Output directory for CSV/plots",
    )
    parser.add_argument(
        "--total-sms",
        type=int,
        default=48,
        help="Total SM count for LDPC adapter conversion",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline training run",
    )
    parser.add_argument(
        "--skip-greenctx",
        action="store_true",
        help="Skip green-context training run",
    )
    parser.add_argument(
        "--dump-gemm-shapes",
        action="store_true",
        help="Dump unique GEMM shapes observed by autograd hooks",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of training runs",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed (incremented per run)",
    )
    parser.add_argument(
        "--max-trace-slots",
        type=int,
        default=27462,
        help=(
            "Truncate training to this many trace slots (shorter trace length)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (legacy alias for --seed-base)",
    )
    parser.add_argument(
        "--skip-violation-analysis",
        action="store_true",
        help="Skip generating per-run violation artifacts",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args()
    _run_benchmark(args)
