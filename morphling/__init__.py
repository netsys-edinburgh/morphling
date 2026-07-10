"""Morphling — DeviceEmulator Python package.

Provides the runtime, hooks, and backend components for distributed
inference device emulation.
"""

try:
    from morphling._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("morphling")
    except Exception:
        __version__ = "0.0.0+unknown"

import morphling
import morphling.hooks as hooks


def set_backend(backend):
    """Set the active compute backend for morphling autograd hooks."""
    from morphling.hooks import autograd

    print(f"Setting backend to {(backend.__class__.__name__)}")
    autograd.set_backend(backend)


__all__ = ["set_backend", "__version__"]
