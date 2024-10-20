from abc import ABC, abstractmethod


class BaseBackend(ABC):
    @abstractmethod
    def on_response(self, message):
        pass


import io

import torch


class MatMulRequestMessage:
    def set(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        r: int,
        c: int,
        block_size: int,
    ):
        self.a = mat_a[r * block_size : (r + 1) * block_size, :]
        self.b = mat_b[:, c * block_size : (c + 1) * block_size]

        # self.c = torch.zeros(block_size, block_size)

        self.r = r
        self.c = c

    def serialize(self) -> bytes:
        a_bytes = io.BytesIO()
        torch.save(self.a, a_bytes)
        a_bytes = a_bytes.getvalue()
        a_byte_len = len(a_bytes)

        b_bytes = io.BytesIO()
        torch.save(self.b, b_bytes)
        b_bytes = b_bytes.getvalue()
        b_byte_len = len(b_bytes)

        body_bytes = (
            self.r.to_bytes(8, byteorder="big")
            + self.c.to_bytes(8, byteorder="big")
            + a_byte_len.to_bytes(8, byteorder="big")
            + b_byte_len.to_bytes(8, byteorder="big")
            + a_bytes
            + b_bytes
        )

        return body_bytes

    def decentralize(self, body: bytes):
        self.r = int.from_bytes(body[:8], byteorder="big")
        self.c = int.from_bytes(body[8:16], byteorder="big")

        a_size = int.from_bytes(body[16:24], byteorder="big")
        b_size = int.from_bytes(body[24:32], byteorder="big")

        a = body[32 : 32 + a_size]
        b = body[32 + a_size :]

        self.a = torch.load(io.BytesIO(a))
        self.b = torch.load(io.BytesIO(b))

    # all functions has async wrapper

    async def async_serialize(self) -> bytes:
        return self.serialize()

    async def async_decentralize(self, body: bytes):
        self.decentralize(body)


class MatMulResponseMessage:
    def set(self, result: torch.Tensor, r: int, c: int):
        self.result = result
        self.r = r
        self.c = c

    def serialize(self) -> bytes:
        result_bytes = io.BytesIO()
        torch.save(self.result, result_bytes)
        result_bytes = result_bytes.getvalue()
        result_byte_len = len(result_bytes)

        body_bytes = (
            self.r.to_bytes(8, byteorder="big")
            + self.c.to_bytes(8, byteorder="big")
            + result_byte_len.to_bytes(8, byteorder="big")
            + result_bytes
        )

        return body_bytes

    def decentralize(self, body: bytes):
        self.r = int.from_bytes(body[:8], byteorder="big")
        self.c = int.from_bytes(body[8:16], byteorder="big")

        result_size = int.from_bytes(body[16:24], byteorder="big")

        result = body[24 : 24 + result_size]

        self.result = torch.load(io.BytesIO(result))

    async def async_serialize(self) -> bytes:
        return self.serialize()

    async def async_decentralize(self, body: bytes):
        return self.decentralize(body)
