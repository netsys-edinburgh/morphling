from __future__ import annotations

import asyncio
import importlib
import json
import threading
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import AsyncContextManager, Optional, Protocol, TypeVar, cast


@dataclass
class BarrierResult:
    """Result of a structural barrier synchronization."""

    phase: str
    step_idx: int
    virtual_times_s: dict[int, float]
    max_virtual_time_s: float
    wall_barrier_time_s: float


class _BarrierImpl(Protocol):
    def barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> BarrierResult: ...


_T = TypeVar("_T")


class _IncomingMessageLike(Protocol):
    body: bytes

    def process(self) -> AsyncContextManager[object]: ...


class _ThreadingBarrierImpl:
    def __init__(self, num_devices: int) -> None:
        self._num_devices: int = num_devices
        self._barrier: threading.Barrier = threading.Barrier(num_devices)
        self._lock: threading.Lock = threading.Lock()
        self._virtual_times: dict[int, float] = {}
        self._phase: str = ""
        self._step_idx: int = 0
        self._result: Optional[BarrierResult] = None

    def barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> BarrierResult:
        wall_start = time.perf_counter()

        with self._lock:
            self._virtual_times[rank] = virtual_time_s
            self._phase = phase
            self._step_idx = step_idx

        elected = self._barrier.wait()

        if elected == 0:
            vt = dict(self._virtual_times)
            self._result = BarrierResult(
                phase=phase,
                step_idx=step_idx,
                virtual_times_s=vt,
                max_virtual_time_s=max(vt.values()),
                wall_barrier_time_s=0.0,
            )

        _ = self._barrier.wait()

        result = self._result
        assert result is not None

        wall_elapsed = time.perf_counter() - wall_start
        per_thread_result = BarrierResult(
            phase=result.phase,
            step_idx=result.step_idx,
            virtual_times_s=result.virtual_times_s,
            max_virtual_time_s=result.max_virtual_time_s,
            wall_barrier_time_s=wall_elapsed,
        )

        if elected == 0:
            _ = self._barrier.wait()
            with self._lock:
                self._virtual_times.clear()
            self._result = None
        else:
            _ = self._barrier.wait()

        return per_thread_result


class _RabbitMQBarrierImpl:
    _JOIN_QUEUE: str = "emulation_barrier_join"
    _RESULT_EXCHANGE: str = "emulation_barrier_result"

    def __init__(
        self,
        num_devices: int,
        host: str = "amqp://localhost/",
        run_id: Optional[str] = None,
        collector_rank: int = 0,
        timeout_s: float = 30.0,
    ) -> None:
        self._num_devices: int = num_devices
        self._host: str = host
        self._run_id: str = run_id or str(uuid.uuid4())
        self._collector_rank: int = collector_rank
        self._timeout_s: float = timeout_s
        self._collector_lock: threading.Lock = threading.Lock()
        self._collector_started: bool = False

    def barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> BarrierResult:
        if rank == self._collector_rank:
            self._ensure_collector_started()

        wall_start = time.perf_counter()
        payload = self._run_async(
            self._async_barrier(rank, phase, step_idx, virtual_time_s)
        )
        wall_elapsed = time.perf_counter() - wall_start

        payload_map = cast(dict[str, object], payload)
        times = cast(dict[str, float], payload_map["virtual_times_s"])
        phase_out = cast(str, payload_map["phase"])
        step_out = cast(int, payload_map["step_idx"])
        max_out = cast(float, payload_map["max_virtual_time_s"])
        vt = {int(k): float(v) for k, v in times.items()}
        return BarrierResult(
            phase=phase_out,
            step_idx=step_out,
            virtual_times_s=vt,
            max_virtual_time_s=max_out,
            wall_barrier_time_s=wall_elapsed,
        )

    def _ensure_collector_started(self) -> None:
        with self._collector_lock:
            if self._collector_started:
                return
            t = threading.Thread(
                target=self._collector_thread_main,
                daemon=True,
            )
            t.start()
            self._collector_started = True

    @staticmethod
    def _run_async(coro: Coroutine[object, object, _T]) -> _T:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _collector_thread_main(self) -> None:
        self._run_async(self._collector_main())

    async def _collector_main(self) -> None:
        aio_pika = importlib.import_module("aio_pika")

        connection = await aio_pika.connect_robust(self._host)
        channel = await connection.channel()

        join_queue = await channel.declare_queue(
            self._JOIN_QUEUE,
            durable=False,
            auto_delete=False,
        )
        result_exchange = await channel.declare_exchange(
            self._RESULT_EXCHANGE,
            aio_pika.ExchangeType.FANOUT,
        )

        pending: dict[tuple[int, str], dict[int, float]] = {}

        async def on_join(message: _IncomingMessageLike) -> None:
            async with message.process():
                data = json.loads(message.body.decode())
                if data.get("run_id") != self._run_id:
                    return

                step_idx = int(data["step_idx"])
                phase = str(data["phase"])
                rank = int(data["rank"])
                vt = float(data["virtual_time_s"])

                key = (step_idx, phase)
                arrivals = pending.setdefault(key, {})
                arrivals[rank] = vt

                if len(arrivals) == self._num_devices:
                    result = {
                        "run_id": self._run_id,
                        "step_idx": step_idx,
                        "phase": phase,
                        "virtual_times_s": {
                            str(r): arrivals[r] for r in sorted(arrivals)
                        },
                        "max_virtual_time_s": max(arrivals.values()),
                    }
                    await result_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(result).encode(),
                            delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
                        ),
                        routing_key="",
                    )
                    del pending[key]

        await join_queue.consume(on_join)
        await asyncio.Future()

    async def _async_barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> dict[str, object]:
        aio_pika = importlib.import_module("aio_pika")

        connection = await aio_pika.connect_robust(self._host)
        channel = await connection.channel()

        join_queue = await channel.declare_queue(
            self._JOIN_QUEUE,
            durable=False,
            auto_delete=False,
        )
        result_exchange = await channel.declare_exchange(
            self._RESULT_EXCHANGE,
            aio_pika.ExchangeType.FANOUT,
        )
        result_queue = await channel.declare_queue(exclusive=True)
        await result_queue.bind(result_exchange)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, object]] = loop.create_future()

        async def on_result(message: _IncomingMessageLike) -> None:
            async with message.process():
                data = json.loads(message.body.decode())
                if data.get("run_id") != self._run_id:
                    return
                if int(data.get("step_idx", -1)) != step_idx:
                    return
                if str(data.get("phase", "")) != phase:
                    return
                if not future.done():
                    future.set_result(data)

        consumer_tag = await result_queue.consume(on_result)

        join_payload = {
            "run_id": self._run_id,
            "rank": rank,
            "step_idx": step_idx,
            "phase": phase,
            "virtual_time_s": virtual_time_s,
        }
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(join_payload).encode(),
                delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
            ),
            routing_key=join_queue.name,
        )

        try:
            result = await asyncio.wait_for(future, timeout=self._timeout_s)
            return result
        finally:
            await result_queue.cancel(consumer_tag)
            await connection.close()


class StructuralBarrier:
    def __init__(
        self,
        num_devices: int,
        backend: str = "threading",
        **kwargs: object,
    ) -> None:
        self._impl: _BarrierImpl
        if backend == "threading":
            self._impl = _ThreadingBarrierImpl(num_devices)
        elif backend == "rabbitmq":
            host = cast(str, kwargs.get("host", "amqp://localhost/"))
            run_id = cast(Optional[str], kwargs.get("run_id"))
            collector_rank = cast(int, kwargs.get("collector_rank", 0))
            timeout_s = cast(float, kwargs.get("timeout_s", 30.0))
            self._impl = _RabbitMQBarrierImpl(
                num_devices=num_devices,
                host=host,
                run_id=run_id,
                collector_rank=collector_rank,
                timeout_s=timeout_s,
            )
        else:
            raise ValueError(f"Unknown barrier backend: {backend}")

    def barrier(
        self,
        rank: int,
        phase: str,
        step_idx: int,
        virtual_time_s: float,
    ) -> BarrierResult:
        return self._impl.barrier(rank, phase, step_idx, virtual_time_s)
