# pyright: reportMissingImports=false
"""Microbenchmark for ScheduleGemmRequest protobuf serialize/deserialize (#50).

Replaces the deleted ``test_real_comparison.py`` FlatBuffers placeholder,
which incorrectly assumed the wire format carries raw matrix bytes. The
actual scheduler RPC uses shared-memory handles (``ShmInfo``); this bench
tracks regression on the real serialized path.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("morphling.proto.morphling_pb2")

from morphling.proto import morphling_pb2 as pb  # noqa: E402


def _make_request(num_shards: int) -> pb.ScheduleGemmRequest:
    req = pb.ScheduleGemmRequest()
    req.task_id = "bench-task-0001"
    req.device_id = 3
    req.sync_stage = 7
    req.task_info.name = "task-shm-0001"
    req.task_info.size = 4096
    for i in range(num_shards):
        a = req.a_info.add()
        a.name = f"a-shard-{i:04d}"
        a.size = 1024 * 1024
        b = req.b_info.add()
        b.name = f"b-shard-{i:04d}"
        b.size = 1024 * 1024
        c = req.c_info.add()
        c.name = f"c-shard-{i:04d}"
        c.size = 1024 * 1024
    return req


@pytest.mark.benchmark
@pytest.mark.parametrize("num_shards", [1, 16, 128])
def test_schedule_gemm_roundtrip(
    num_shards: int, capsys: pytest.CaptureFixture[str]
) -> None:
    req = _make_request(num_shards)

    start = time.perf_counter()
    buf = req.SerializeToString()
    ser_us = (time.perf_counter() - start) * 1e6

    decoded = pb.ScheduleGemmRequest()
    start = time.perf_counter()
    decoded.ParseFromString(buf)
    deser_us = (time.perf_counter() - start) * 1e6

    assert decoded.task_id == req.task_id
    assert decoded.device_id == req.device_id
    assert decoded.sync_stage == req.sync_stage
    assert len(decoded.a_info) == num_shards
    assert len(decoded.b_info) == num_shards
    assert len(decoded.c_info) == num_shards
    if num_shards:
        assert decoded.a_info[0].name == req.a_info[0].name

    with capsys.disabled():
        print(
            f"\n  shards={num_shards}  serialize={ser_us:.2f}us  "
            f"deserialize={deser_us:.2f}us  payload={len(buf)}B"
        )
