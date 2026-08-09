"""Command-line entry point for the coordinator-scaling figure.

Every input path is required and must point at an existing file; a missing or
malformed input fails clearly with a non-zero exit code and a message naming
the offending file rather than emitting a partial figure.  The figure is never
generated from placeholder data -- the caller supplies the real
``summary.json`` / ``strong.json`` / ``weak.json`` / ``breakdown.json``.

Usage::

    python3 -m scripts.coord_scaling_plot \\
        --summary  results/coord_scaling/single/summary.json \\
        --strong   results/coord_scaling/strong.json \\
        --weak     results/coord_scaling/weak.json \\
        --breakdown results/coord_scaling/breakdown.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from . import schema

_DEFAULT_OUT = "results/coord_scaling/figures/coordinator_scaling"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coord_scaling_plot",
        description="Render the three-panel coordinator-scaling paper figure.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        type=Path,
        help="single-coordinator device-scaling summary.json",
    )
    parser.add_argument(
        "--strong", required=True, type=Path, help="strong-scaling strong.json"
    )
    parser.add_argument(
        "--weak", required=True, type=Path, help="weak-scaling weak.json"
    )
    parser.add_argument(
        "--breakdown",
        required=True,
        type=Path,
        help="per-iteration breakdown.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(_DEFAULT_OUT),
        help="output path stem (a .pdf and .png are written); "
        f"default: {_DEFAULT_OUT}",
    )
    return parser


def _load_inputs(args: argparse.Namespace):
    device = schema.parse_device_scaling(schema.load_json(args.summary))
    strong = schema.parse_scaling_efficiency(
        schema.load_json(args.strong), "strong"
    )
    weak = schema.parse_scaling_efficiency(schema.load_json(args.weak), "weak")
    breakdown = schema.parse_breakdown(schema.load_json(args.breakdown))
    return device, strong, weak, breakdown


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        device, strong, weak, breakdown = _load_inputs(args)
    except schema.SchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from . import (
        figure,  # imported lazily so schema-only errors avoid matplotlib
    )

    pdf_path, png_path = figure.render(
        device, strong, weak, breakdown, args.out
    )
    _report(pdf_path, png_path)
    return 0


def _report(pdf_path: Path, png_path: Path) -> Tuple[Path, Path]:
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    return pdf_path, png_path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
