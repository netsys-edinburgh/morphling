from __future__ import annotations

def _parse_trace_file(
    path: str,
) -> tuple[list[tuple[int, int]], str, str]:
    entries: list[tuple[int, int]] = []
    time_unit = "us"
    clock_mode = "step"
    with open(path) as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                directive = line.lstrip("#").strip()
                if directive.startswith("time_unit="):
                    time_unit = directive.split("=", 1)[1]
                elif directive.startswith("clock_mode="):
                    clock_mode = directive.split("=", 1)[1]
                continue
            if line.startswith("time_unit="):
                time_unit = line.split("=", 1)[1]
                continue
            if line.startswith("clock_mode="):
                clock_mode = line.split("=", 1)[1]
                continue
            parts = line.split(",")
            if not header_skipped:
                try:
                    _ = int(parts[0])
                except ValueError:
                    header_skipped = True
                    continue
                header_skipped = True
            if len(parts) >= 2:
                try:
                    ts = int(parts[0])
                    sms = int(parts[1])
                    entries.append((ts, sms))
                except ValueError:
                    continue
    return entries, time_unit, clock_mode


def _sm_count_at_step(
    entries: list[tuple[int, int]],
    step: int,
    default_sm: int,
) -> int:
    result = default_sm
    for ts, sms in entries:
        if ts <= step:
            result = sms
        else:
            break
    return result


def _sm_count_at_time(
    entries: list[tuple[int, int]],
    elapsed_us: int,
    time_unit: str,
    default_sm: int,
) -> int:
    if time_unit == "s":
        scale = 1_000_000
    elif time_unit == "ms":
        scale = 1_000
    else:
        scale = 1
    result = default_sm
    for ts, sms in entries:
        threshold_us = ts * scale
        if elapsed_us >= threshold_us:
            result = sms
        else:
            break
    return result


parse_trace_file = _parse_trace_file
sm_count_at_step = _sm_count_at_step
sm_count_at_time = _sm_count_at_time
