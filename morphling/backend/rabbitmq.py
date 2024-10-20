import asyncio
import uuid

import aio_pika
import numpy as np
import torch

from .base import MatMulRequestMessage, MatMulResponseMessage


class RabbitMQBackend:
    def __init__(self, loop, host="amqp://localhost/"):
        self.host = host
        self.loop = loop
        self.connection = None
        self.channel = None

    async def connect(self):
        self.connection = await aio_pika.connect_robust(
            self.host, loop=self.loop
        )
        self.channel = await self.connection.channel()

        await self.channel.set_qos(prefetch_count=1)
        # Declare the queues and keep references to them
        self.request_queue = await self.channel.declare_queue(
            "mm_request_queue", durable=True
        )
        self.response_queue = await self.channel.declare_queue(
            "mm_response_queue", durable=True
        )

        # Set up the consumer using the correct method
        await self.response_queue.consume(self.on_response)

    async def on_response(self, message: aio_pika.IncomingMessage):
        async with message.process():
            response = MatMulResponseMessage()
            await response.async_decentralize(message.body)

            r, c = response.r, response.c
            print(f"Received block ({r}, {c})")
            self.c[
                r * self.block_size : (r + 1) * self.block_size,
                c * self.block_size : (c + 1) * self.block_size,
            ] = response.result

            if torch.isnan(self.c).sum() == 0:
                self.finished = True

    def create_output_matrix(self, in_dim, out_dim, block_size):
        assert in_dim % block_size == 0
        assert out_dim % block_size == 0

        self.c = torch.zeros(in_dim, out_dim)
        self.block_size = block_size
        self.c[:] = float("nan")
        self.finished = False

    async def call(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        r: int,
        c: int,
        block_size: int,
    ):
        request = MatMulRequestMessage()
        request.set(mat_a, mat_b, r, c, block_size)
        message = await request.async_serialize()
        corr_id = str(uuid.uuid4())

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=message,
                reply_to="mm_response_queue",
                correlation_id=corr_id,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key="mm_request_queue",
        )


class RabbitMQWorker:
    def __init__(self, cid, loop, host="amqp://localhost/", emulation=False):
        self.cid = cid
        self.loop = loop
        self.host = host
        self.connection = None
        self.channel = None
        self.emulation = emulation

        if emulation:
            # enable interception of torch.mm
            import morphling._C

    async def connect(self):
        self.connection = await aio_pika.connect_robust(
            self.host, loop=self.loop
        )
        self.channel = await self.connection.channel()
        await self.setup_queues()

    async def setup_queues(self):
        # Declare queues
        self.request_queue = await self.channel.declare_queue(
            "mm_request_queue", durable=True
        )
        await self.channel.declare_queue("mm_response_queue", durable=True)

        # Set Quality of Service
        await self.channel.set_qos(prefetch_count=1)

    async def start_consuming(self):
        await self.request_queue.consume(self.on_request)
        print(f" [x] {self.cid}: Waiting for RPC requests")
        await asyncio.Future()  # Run forever

    async def on_request(self, message: aio_pika.IncomingMessage):
        async with message.process():
            request = MatMulRequestMessage()
            await request.async_decentralize(message.body)

            result = await self.matmul(request.a, request.b)
            print(f"[x] {self.cid}: {result.size()}")

            response = MatMulResponseMessage()
            response.set(result, request.r, request.c)
            message_body = await response.async_serialize()

            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    correlation_id=message.correlation_id,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=message.reply_to,
            )
            print(f"[x] {self.cid}: Sent response")

    async def matmul(self, a, b):
        if self.emulation:
            # Emulate the device
            return torch.mm(a, b)

        # Check for MPS or CUDA availability and use the appropriate device
        device = "mps" if torch.backends.mps.is_available() else "cuda:0"
        a = a.to(device)
        b = b.to(device)
        return torch.mm(a, b).to("cpu")
