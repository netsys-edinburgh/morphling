"""Morphling — DeviceEmulator Python package.

Provides the runtime, hooks, and backend components for distributed
inference device emulation.
"""

import morphling
import morphling.hooks as hooks


def set_backend(backend):
    print(f"Setting backend to {(backend.__class__.__name__)}")
    morphling.hooks.autograd._backend = backend
