from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

ROLE_NAMES = ("compute", "recv", "send", "dp")


@dataclass
class StreamBundle:
    comp: object
    recv: object
    send: object
    dp: object
    sm_count: int
    generation: int

    def streams_dict(self) -> dict[str, object]:
        return {
            "compute": self.comp,
            "recv": self.recv,
            "send": self.send,
            "dp": self.dp,
        }


class GreenCtxBackend(Protocol):
    def is_supported(self) -> bool: ...

    def unsupported_reason(self) -> str: ...

    def available_sm_counts(self) -> list[int]: ...

    def sm_step(self) -> int: ...

    def partition_sm_count(self) -> int: ...

    def get_stream_bundle(self, sm_count: int) -> StreamBundle: ...

    def activate_for_step(self, step_or_time: int) -> tuple[int, int]: ...

    def activate_for_time(self, elapsed_us: int) -> tuple[int, int]: ...

    def deactivate(self, prev_sm_count: int) -> None: ...

    def load_trace(self, path: str) -> bool: ...

    def sm_count_at_time(self, elapsed_us: int) -> int: ...

    def sm_count_at_step(self, step: int) -> int: ...

    def switch_count(self) -> int: ...

    def close(self) -> None: ...
