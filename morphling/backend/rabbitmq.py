import asyncio
import time
import uuid

import aio_pika
import numpy as np
import torch

from .base import BaseBackend, MatMulRequestMessage, MatMulResponseMessage


class RabbitMQBackend(BaseBackend):
    def __init__(self, loop, host="amqp://localhost/", block_size=32):
        self.host = host
        self.loop = loop
        self.connection = None
        self.channel = None
        self.block_size = block_size

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

        return self

    # def connect(self):
    #     self.connection = pika.BlockingConnection(pika.URLParameters(self.host))
    #     self.channel = self.connection.channel()

    #     self.channel.queue_declare(queue="mm_request_queue", durable=True)
    #     self.channel.queue_declare(queue="mm_response_queue", durable=True)

    #     self.channel.basic_qos(prefetch_count=1)
    #     self.channel.basic_consume(queue="mm_response_queue", on_message_callback=self.on_response, auto_ack=True)

    #     return self

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

    def create_output_matrix(self, mat_a: torch.Tensor, mat_b: torch.Tensor):
        # check matrix number of dimensions:
        # 1) mat_a and mat_b are 2D matrices
        # 2) mat_a > 2D and mat_b is 2D
        # 3) mat_a > 2D and mat_b > 2D

        a_shape = mat_a.size()
        b_shape = mat_b.size()

        in_dim = a_shape[-2]
        out_dim = b_shape[-1]

        assert (
            in_dim % self.block_size == 0
        ), f"Input dimension {in_dim} must be divisible by block size {self.block_size}"
        assert (
            out_dim % self.block_size == 0
        ), f"Output dimension {out_dim} must be divisible by block size {self.block_size}"

        if len(a_shape) == 2 and len(b_shape) == 2:
            c_shape = (in_dim, out_dim)
            ld_list
        elif len(a_shape) > 2 and len(b_shape) == 2:
            c_shape = a_shape[:-2] + (in_dim, out_dim)
        else:
            lda_shape = a_shape[:-2]
            ldb_shape = b_shape[:-2]
            # ldx dim must be the same
            assert (
                lda_shape == ldb_shape
            ), f"Input dimensions {lda_shape} and {ldb_shape} must be the same"
            c_shape = a_shape[:-2] + (in_dim, out_dim)

        self.c = torch.empty(c_shape)
        self.c[:] = float("nan")
        self.finished = False

    async def single_matmul_call(self, mat_a, mat_b, r, c):
        request = MatMulRequestMessage()
        request.set(mat_a, mat_b, r, c, self.block_size)
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

    async def call(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        r: int,
        c: int,
    ):
        # check matrix number of dimensions:
        # 1) mat_a and mat_b are 2D matrices
        # 2) mat_a > 2D and mat_b is 2D
        # 3) mat_a > 2D and mat_b > 2D

        if Len(mat_a.size()) == 2 and len(mat_b.size()) == 2:
            await self.single_matmul_call(mat_a, mat_b, r, c)

        elif len(mat_a.size()) > 2 and len(mat_b.size()) == 2:
            for i in range(mat_a.size(0)):
                await self.single_matmul_call(mat_a[i], mat_b, r, c)
        else:
            for i in range(mat_a.size(0)):
                for j in range(mat_b.size(0)):
                    await self.call(mat_a[i], mat_b[j], r, c)

    def dispatch_matmul(
        self, mat_a: torch.Tensor, mat_b: torch.Tensor
    ) -> torch.Tensor:
        self.create_output_matrix(mat_a, mat_b)

        for r in range(in_dim // block_size):
            for c in range(out_dim // block_size):
                asyncio.ensure_future(self.call(mat_a, mat_b, r, c))

        while not self.finished:
            time.sleep(0.1)

        return self.c


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
