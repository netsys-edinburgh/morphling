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
except Exception:
    EmulationEngine = None  # type: ignore
    InitEmptyModel = None  # type: ignore
