from __future__ import annotations

import warnings

import torch
import torch.cuda

from .base import StreamBundle


class OffBackend:
    def __init__(self, gpu_id: int = 0, strict: bool = False):
        self._gpu_id: int = gpu_id
        self._strict: bool = strict
        self._closed: bool = False

        if strict:
            raise RuntimeError(
                "Green contexts requested (strict=True) but no backend is available"
            )

        with torch.cuda.device(gpu_id):
            self._streams: dict[str, object] = {
                "compute": torch.cuda.Stream(device=gpu_id),
                "recv": torch.cuda.Stream(device=gpu_id),
                "send": torch.cuda.Stream(device=gpu_id),
                "dp": torch.cuda.Stream(device=gpu_id),
            }
        self._bundle: StreamBundle = StreamBundle(
            comp=self._streams["compute"],
            recv=self._streams["recv"],
            send=self._streams["send"],
            dp=self._streams["dp"],
            sm_count=0,
            generation=0,
        )

    def is_supported(self) -> bool:
        return False

    def unsupported_reason(self) -> str:
        return "Off backend (no green context)"

    def available_sm_counts(self) -> list[int]:
        return [0]

    def sm_step(self) -> int:
        return 0

    def partition_sm_count(self) -> int:
        return 0

    def get_stream_bundle(self, sm_count: int) -> StreamBundle:
        _ = sm_count
        return self._bundle

    def activate_for_step(self, step_or_time: int) -> tuple[int, int]:
        _ = step_or_time
        return 0, 0

    def activate_for_time(self, elapsed_us: int) -> tuple[int, int]:
        _ = elapsed_us
        return 0, 0

    def deactivate(self, prev_sm_count: int) -> None:
        _ = prev_sm_count
        pass

    def load_trace(self, path: str) -> bool:
        _ = path
        warnings.warn(
            "OffBackend: trace ignored (no green context)",
            stacklevel=2,
        )
        return True

    def sm_count_at_time(self, elapsed_us: int) -> int:
        _ = elapsed_us
        return 0

    def sm_count_at_step(self, step: int) -> int:
        _ = step
        return 0

    def switch_count(self) -> int:
        return 0

    def close(self) -> None:
        self._closed = True
