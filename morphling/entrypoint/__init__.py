"""Entry point modules for Morphling device emulator.

Provides CLI interfaces and device configuration utilities.
"""

from .generate_device_config import DeviceConfigArguments, ModelConfigArguments

__all__ = ["DeviceConfigArguments", "ModelConfigArguments"]
