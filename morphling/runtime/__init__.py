"""Morphling runtime components for device emulation.

Provides:
  - EmulationEngine: Core model emulation engine for distributed inference
  - InitEmptyModel: Context manager for initializing empty models
  - GreenContextController: Trace-driven SM partitioning controller
  - LdpcTraceAdapter: Adapter for LDPC trace data

These components enable device and network emulation for distributed
inference workflows with optional green context (CUDA) support.
"""

try:
    from .model_emulator import EmulationEngine, InitEmptyModel
except ImportError as e:
    import warnings

    warnings.warn(
        "Failed to import EmulationEngine (C++ bindings not available): "
        + f"{e}. Some functionality will be disabled.",
        ImportWarning,
        stacklevel=2,
    )
    EmulationEngine = None
    InitEmptyModel = None


__all__ = ["EmulationEngine", "InitEmptyModel"]
