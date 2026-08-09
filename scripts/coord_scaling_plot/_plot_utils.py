"""Locate and adapt the conference-plot skill's ``plot_utils`` helpers.

The paper style, Wong colorblind-safe palette and dual-output saver live in the
``conference-plot`` skill outside this repository.  This shim finds that module
(via the ``CONFERENCE_PLOT_SCRIPTS`` environment variable, else the known skill
location), forces a headless ``Agg`` backend before matplotlib initializes, and
re-exports the pieces the figure builder needs.

It also adds :func:`save_paper_figure`, which writes a vector PDF through the
skill's ``save_dual_output`` and a 300-dpi PNG directly -- the skill helper
emits PNGs at 150 dpi, so the raster path is written here to meet the paper's
300-dpi requirement.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402  (backend must be set before pyplot import)

matplotlib.use("Agg", force=True)

_DEFAULT_SCRIPTS = Path(
    "/home/xly/.opencode/skills/conference-plot/scripts/plot_utils.py"
)


def _load_plot_utils():
    override = os.environ.get("CONFERENCE_PLOT_SCRIPTS")
    candidate = (
        Path(override) / "plot_utils.py" if override else _DEFAULT_SCRIPTS
    )
    if not candidate.is_file():
        raise ImportError(
            "conference-plot plot_utils.py not found at "
            f"{candidate}; set CONFERENCE_PLOT_SCRIPTS to its scripts directory"
        )
    spec = importlib.util.spec_from_file_location(
        "conference_plot_plot_utils", candidate
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load a module spec from {candidate}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_plot_utils = _load_plot_utils()

paper_style = _plot_utils.paper_style
save_dual_output = _plot_utils.save_dual_output
WONG_PALETTE = _plot_utils.WONG_PALETTE
HATCHES = _plot_utils.HATCHES


def save_paper_figure(fig, out_base: Path) -> Tuple[Path, Path]:
    """Write ``out_base`` as a vector PDF and a 300-dpi PNG; return both paths."""
    base = Path(out_base)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    save_dual_output(fig, pdf_path, None, save_both=False)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    return pdf_path, png_path
