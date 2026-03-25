import asyncio
from unittest.mock import AsyncMock, patch

import pytest

torch = pytest.importorskip("torch")
base_mod = pytest.importorskip("morphling.backend.base")
BaseBackend = base_mod.BaseBackend
MatMulRequestMessage = base_mod.MatMulRequestMessage


def test_base_backend_is_abstract_and_requires_dispatch_implementation():
    with pytest.raises(TypeError):
        _ = BaseBackend()  # pyright: ignore[reportAbstractUsage]


def test_matmul_request_message_round_trip_serialization():
    mat_a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mat_b = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    request = MatMulRequestMessage()
    request.set(mat_a, mat_b, r=1, c=1, block_size=2, ld=[7])

    payload = request.serialize()

    decoded = MatMulRequestMessage()
    decoded.deserialize(payload)

    assert decoded.r == 1
    assert decoded.c == 1
    assert list(decoded.ld) == [7]
    assert torch.equal(decoded.a, mat_a[2:4, :])
    assert torch.equal(decoded.b, mat_b[:, 2:4])


def test_rabbitmq_backend_connect_and_disconnect_without_live_server():
    rabbitmq_mod = pytest.importorskip("morphling.backend.rabbitmq")
    RabbitMQBackend = rabbitmq_mod.RabbitMQBackend

    loop = asyncio.new_event_loop()
    backend = RabbitMQBackend(
        loop=loop, host="amqp://unit-test/", block_size=16
    )

    request_queue = AsyncMock()
    response_queue = AsyncMock()
    timer_sync_response_queue = AsyncMock()
    timer_sync_request_exchange = AsyncMock()
    timer_sync_set_exchange = AsyncMock()

    channel = AsyncMock()
    channel.set_qos = AsyncMock()
    channel.declare_queue = AsyncMock(
        side_effect=[
            request_queue,
            response_queue,
            timer_sync_response_queue,
        ]
    )
    channel.declare_exchange = AsyncMock(
        side_effect=[
            timer_sync_request_exchange,
            timer_sync_set_exchange,
        ]
    )

    connection = AsyncMock()
    connection.channel = AsyncMock(return_value=channel)
    connection.close = AsyncMock()

    connect_mock = AsyncMock(return_value=connection)
    with patch(
        "morphling.backend.rabbitmq.aio_pika.connect_robust",
        new=connect_mock,
    ):
        connected = asyncio.run(backend.connect())

    assert connected is backend
    assert backend.connection is connection
    assert backend.channel is channel

    connect_mock.assert_awaited_once_with("amqp://unit-test/", loop=loop)
    channel.set_qos.assert_awaited_once_with(prefetch_count=1)
    response_queue.consume.assert_awaited_once_with(backend.on_response)
    timer_sync_response_queue.consume.assert_awaited_once_with(
        backend.on_timer_rsp
    )

    assert backend.connection is not None
    asyncio.run(backend.connection.close())
    connection.close.assert_awaited_once()


def test_rabbitmq_backend_connect_propagates_connection_errors():
    rabbitmq_mod = pytest.importorskip("morphling.backend.rabbitmq")
    RabbitMQBackend = rabbitmq_mod.RabbitMQBackend

    loop = asyncio.new_event_loop()
    backend = RabbitMQBackend(loop=loop, host="amqp://broken-host/")

    connect_mock = AsyncMock(side_effect=RuntimeError("connect failed"))
    with patch(
        "morphling.backend.rabbitmq.aio_pika.connect_robust",
        new=connect_mock,
    ):
        with pytest.raises(RuntimeError, match="connect failed"):
            asyncio.run(backend.connect())

    assert backend.connection is None
    assert backend.channel is None
