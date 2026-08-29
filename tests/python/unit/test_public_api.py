"""Public API surface checks.

The import runs in a fresh interpreter: pytest's in-process import machinery can
leave ``morphling`` submodules half-initialized across the full suite (the image
carries both an installed copy and the source tree), whereas downstream
consumers import the API in their own process. This verifies that path directly.
"""

import subprocess
import sys

_SURFACE = (
    "set_backend AutoBackend apply_hooks LinearFunction DeviceConfigArguments "
    "ModelConfigArguments add_metrics_arguments metrics_config_from_args "
    "start_metrics_collector CoordinatorMetricsCollector PhaseRecorder "
    "track_phase GreenContextRuntime"
).split()


def _run(code: str) -> None:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr


def test_public_api_surface_is_importable():
    _run(
        "import morphling.api as a\n"
        f"expected = set({_SURFACE!r})\n"
        "missing = expected - set(a.__all__)\n"
        "assert not missing, f'missing from __all__: {sorted(missing)}'\n"
        "[getattr(a, n) for n in expected]\n"
    )


def test_greencontextruntime_public_alias_matches_compiled():
    _run(
        "from morphling._GreenCtx import GreenContextRuntime as c\n"
        "from morphling.runtime import GreenContextRuntime as p\n"
        "assert p is c\n"
    )
