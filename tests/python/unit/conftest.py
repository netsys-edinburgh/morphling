"""Collection guards for coordinator-evaluation unit tests.

The coordinator-evaluation tests import heavy, optional dependencies at module
scope: ``psutil`` (coordinator metrics), ``matplotlib`` and the external
conference-plot helper (scaling figure), and the C-extension-backed
``morphling`` runtime. pytest imports every module during collection before it
can read markers, so these imports would abort the CPU smoke job -- which
installs only ``pytest``/``torch``/``numpy`` -- even though the tests are not
smoke-marked and are never selected there.

``psutil`` is present in every environment that can actually run these tests
(the full ``requirements.txt`` install used by the Docker test image), and
absent from the minimal smoke environment, so it is a reliable signal. When it
is missing we skip collecting these files; full environments collect and run
them normally.
"""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("psutil") is None:
    collect_ignore_glob += [
        "test_coord_scaling_*.py",
        "test_coordinator_*.py",
        "test_multi_coordinator_*.py",
        "test_single_coordinator_*.py",
        "test_run_coordinator_scaling.py",
    ]
