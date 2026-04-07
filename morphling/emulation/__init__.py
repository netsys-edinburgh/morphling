"""Hybrid emulation runtime with virtual time dilation.

Complements morphling.simulator (analytical) with execution-driven estimation.
Real compute and communication execute on hardware; virtual timestamps are
dilated per-device and per-phase for end-to-end performance estimation.
"""

from .barrier import BarrierResult, StructuralBarrier
from .clock import DilatedDeviceClock, PhaseTimingRecord
from .comm_shm import ShmAllReduceCommFn
from .dilation import DeviceDilation, DilationMatrix
from .recorder import EmulationRecorder
from .step_runner import BSPStepRunner, EmulationConfig, EmulationStepResult

__all__ = [
    "DeviceDilation",
    "DilationMatrix",
    "DilatedDeviceClock",
    "PhaseTimingRecord",
    "StructuralBarrier",
    "BarrierResult",
    "ShmAllReduceCommFn",
    "BSPStepRunner",
    "EmulationConfig",
    "EmulationStepResult",
    "EmulationRecorder",
]
