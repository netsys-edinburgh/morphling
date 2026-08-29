"""morphling.api — supported public integration surface.

Stable, versioned entry points for downstream systems (for example Wasp,
https://github.com/netsys-edinburgh/wasp) that build on Morphling's backend,
autograd hooks, coordinator telemetry, and green-context runtime. Import from
this module instead of reaching into submodules directly; only these symbols
carry a stability guarantee across minor releases.
"""

from morphling import set_backend
from morphling.backend import AutoBackend
from morphling.entrypoint import DeviceConfigArguments, ModelConfigArguments
from morphling.hooks import apply_hooks
from morphling.hooks.autograd import LinearFunction
from morphling.runtime import GreenContextRuntime
from morphling.runtime.coordinator_metrics import (
    PhaseRecorder,
    add_metrics_arguments,
    track_phase,
)

__all__ = [
    "set_backend",
    "AutoBackend",
    "apply_hooks",
    "LinearFunction",
    "DeviceConfigArguments",
    "ModelConfigArguments",
    "add_metrics_arguments",
    "PhaseRecorder",
    "track_phase",
    "GreenContextRuntime",
]
