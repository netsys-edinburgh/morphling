import asyncio
import multiprocessing
import uuid
from multiprocessing.managers import DictProxy
from typing import Any, cast

import pytest

from morphling.emulation.barrier import StructuralBarrier

aio_pika = pytest.importorskip("aio_pika")

pytestmark = pytest.mark.rabbitmq


def _worker_single(
    rank: int,
    host: str,
    run_id: str,
    results: DictProxy[int, dict[str, object]],
) -> None:
    barrier = StructuralBarrier(
        2,
        backend="rabbitmq",
        host=host,
        run_id=run_id,
    )
    vt = 0.0 if rank == 0 else 0.01
    br = barrier.barrier(rank, "compute", 0, vt)
    results[rank] = {
        "phase": br.phase,
        "step_idx": br.step_idx,
        "virtual_times_s": br.virtual_times_s,
        "max_virtual_time_s": br.max_virtual_time_s,
    }


def _worker_reusable(
    rank: int,
    host: str,
    run_id: str,
    results: DictProxy[int, list[tuple[str, int, float]]],
) -> None:
    barrier = StructuralBarrier(
        2,
        backend="rabbitmq",
        host=host,
        run_id=run_id,
    )
    out = []
    for phase, step_idx, vt in (
        ("compute", 0, float(rank)),
        ("comm", 1, float(rank + 1)),
    ):
        br = barrier.barrier(rank, phase, step_idx, vt)
        out.append((br.phase, br.step_idx, br.max_virtual_time_s))
    results[rank] = out


def _rabbitmq_ready(host: str) -> bool:
    async def _check() -> bool:
        module = cast(Any, aio_pika)
        try:
            conn = await module.connect_robust(host)
            await conn.close()
            return True
        except Exception:
            return False

    return asyncio.run(_check())


def test_rabbitmq_barrier_2_processes():
    host = "amqp://localhost/"
    if not _rabbitmq_ready(host):
        pytest.skip("RabbitMQ is not reachable on amqp://localhost/")

    manager = multiprocessing.Manager()
    results = cast(DictProxy[int, dict[str, object]], manager.dict())
    run_id = str(uuid.uuid4())

    p0 = multiprocessing.Process(
        target=_worker_single,
        args=(0, host, run_id, results),
    )
    p1 = multiprocessing.Process(
        target=_worker_single,
        args=(1, host, run_id, results),
    )
    p0.start()
    p1.start()
    p0.join(20)
    p1.join(20)

    assert p0.exitcode == 0
    assert p1.exitcode == 0
    assert len(results) == 2

    r0 = results[0]
    r1 = results[1]
    assert r0["phase"] == "compute"
    assert r1["phase"] == "compute"
    assert r0["step_idx"] == 0
    assert r1["step_idx"] == 0
    assert r0["max_virtual_time_s"] == pytest.approx(0.01)
    assert r1["max_virtual_time_s"] == pytest.approx(0.01)
    vt0 = cast(dict[int, float], r0["virtual_times_s"])
    vt1 = cast(dict[int, float], r1["virtual_times_s"])
    assert vt0 == pytest.approx({0: 0.0, 1: 0.01})
    assert vt1 == pytest.approx({0: 0.0, 1: 0.01})
    manager.shutdown()


def test_rabbitmq_barrier_reusable():
    host = "amqp://localhost/"
    if not _rabbitmq_ready(host):
        pytest.skip("RabbitMQ is not reachable on amqp://localhost/")

    manager = multiprocessing.Manager()
    results = cast(DictProxy[int, list[tuple[str, int, float]]], manager.dict())
    run_id = str(uuid.uuid4())

    p0 = multiprocessing.Process(
        target=_worker_reusable,
        args=(0, host, run_id, results),
    )
    p1 = multiprocessing.Process(
        target=_worker_reusable,
        args=(1, host, run_id, results),
    )
    p0.start()
    p1.start()
    p0.join(20)
    p1.join(20)

    assert p0.exitcode == 0
    assert p1.exitcode == 0
    assert len(results) == 2

    r0 = list(results[0])
    r1 = list(results[1])
    assert len(r0) == 2
    assert len(r1) == 2
    assert r0[0][0] == "compute"
    assert r0[1][0] == "comm"
    assert r1[0][0] == "compute"
    assert r1[1][0] == "comm"
    assert r0[0][1] == 0
    assert r0[1][1] == 1
    assert r1[0][1] == 0
    assert r1[1][1] == 1
    manager.shutdown()
