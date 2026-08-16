"""Cross-language contract test for the dynamic green-context shm channel.

A Python producer writes the shm segment using the documented SmNotifyHeader /
SmNotifySlot binary layout and seqlock protocol; the C++ consumer (the real
`ShmSmTargetSource`, exercised via the `sm_notify_probe` helper) reads it back.
This proves a foreign-language scheduler is byte-compatible with the C++ proxy
without standing up the full proxy server.
"""

from __future__ import annotations

import mmap
import os
import struct
import subprocess
from pathlib import Path

import pytest

_MAGIC = 0x534D4743  # 'SMGC'
_VERSION = 1
# Header: magic, version, num_slots, then reserved[13] uint32 -> 64 bytes.
_HDR = struct.Struct("<III52x")
# Slot: seq, gpu_id, partition_idx, target_sms, generation, then pad -> 64 bytes.
_SLOT = struct.Struct("<IiiiQ40x")


def _probe_bin() -> str | None:
    root = Path(__file__).resolve().parents[3]
    build = root / "tests" / "cpp" / "build"
    direct = build / "sm_notify_probe"
    if direct.exists() and os.access(direct, os.X_OK):
        return str(direct)
    for cand in build.rglob("sm_notify_probe"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _shm_path(name: str) -> str:
    return "/dev/shm/" + name.lstrip("/")


def _write_segment(path: str, slots: list[tuple[int, int, int]]):
    size = _HDR.size + _SLOT.size * len(slots)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    mm = mmap.mmap(fd, size)
    mm[0 : _HDR.size] = _HDR.pack(_MAGIC, _VERSION, len(slots))
    for i, (gpu, part, target) in enumerate(slots):
        off = _HDR.size + _SLOT.size * i
        mm[off : off + _SLOT.size] = _SLOT.pack(0, gpu, part, target, 0)
    mm.flush()
    return fd, mm


def _probe(bin_path: str, name: str, gpu: int, part: int) -> str:
    return subprocess.check_output(
        [bin_path, name, str(gpu), str(part)], text=True
    ).strip()


def test_python_producer_cpp_consumer_contract() -> None:
    probe = _probe_bin()
    if probe is None:
        pytest.skip("sm_notify_probe helper not built")

    name = f"/morphling_smcontract_{os.getpid()}"
    path = _shm_path(name)
    fd, mm = _write_segment(path, [(0, 0, 42), (1, 0, 99)])
    try:
        # Each worker reads its own (gpu_id, partition_idx) slot.
        assert _probe(probe, name, 0, 0) == "42"
        assert _probe(probe, name, 1, 0) == "99"

        # A slot the producer never published => kNoChange (-1).
        assert _probe(probe, name, 5, 5) == "-1"

        # Seqlock publish of a new target for slot 0: seq odd, write, seq even.
        off = _HDR.size  # slot 0
        struct.pack_into("<I", mm, off, 1)
        struct.pack_into("<i", mm, off + 12, 24)  # target_sms field offset
        struct.pack_into("<Q", mm, off + 16, 1)  # generation
        struct.pack_into("<I", mm, off, 2)
        mm.flush()
        assert _probe(probe, name, 0, 0) == "24"
    finally:
        mm.close()
        os.close(fd)
        try:
            os.unlink(path)
        except OSError:
            pass
