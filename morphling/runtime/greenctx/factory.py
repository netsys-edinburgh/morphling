from __future__ import annotations

import logging
from typing import Any

from .base import GreenCtxBackend
from .cpp_backend import CppBackend
from .off_backend import OffBackend
from .torch_backend import TorchNativeBackend

logger = logging.getLogger(__name__)


def select_backend(
    backend: str = "auto",
    gpu_id: int = 0,
    strict: bool = False,
    **kwargs: Any,
) -> GreenCtxBackend:
    if backend == "cpp" or backend == "auto":
        try:
            b = CppBackend(gpu_id=gpu_id, strict=strict, **kwargs)
            if b.is_supported():
                logger.info("Using CppBackend for green ctx")
                return b
            if backend == "cpp":
                if strict:
                    raise RuntimeError(
                        f"CppBackend not supported: {b.unsupported_reason()}"
                    )
                logger.warning(
                    "CppBackend requested but not supported: %s. Falling back.",
                    b.unsupported_reason(),
                )
                b.close()
        except ImportError:
            if backend == "cpp":
                raise
            logger.debug(
                "morphling._GreenCtx not available, trying next backend"
            )

    if backend == "torch_native" or backend == "auto":
        b = TorchNativeBackend(gpu_id=gpu_id, strict=strict, **kwargs)
        if b.is_supported():
            logger.info("Using TorchNativeBackend for green ctx")
            return b
        if backend == "torch_native":
            if strict:
                raise RuntimeError(
                    f"TorchNativeBackend not supported: "
                    f"{b.unsupported_reason()}"
                )
            logger.warning(
                "TorchNativeBackend requested but not supported: %s",
                b.unsupported_reason(),
            )

    if backend == "off" or backend == "auto":
        logger.info("Using OffBackend (no green context)")
        return OffBackend(gpu_id=gpu_id, strict=strict)

    raise ValueError(
        f"Unknown backend '{backend}'. Expected: auto|cpp|torch_native|off"
    )
