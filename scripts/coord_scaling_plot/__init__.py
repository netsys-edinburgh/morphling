"""Coordinator-scaling figure pipeline (paper figure D8).

Typed, boundary-validated data-prep (:mod:`schema`) plus a three-panel
double-column figure builder (:mod:`figure`) and a CLI (:mod:`cli`) that
consumes the real ``summary.json`` / ``strong.json`` / ``weak.json`` /
``breakdown.json`` emitted by the coordinator-scaling experiments.

The data-prep layer is intentionally free of any matplotlib dependency so it
can be unit-tested in isolation; import :mod:`figure` only when rendering.
"""

from __future__ import annotations
