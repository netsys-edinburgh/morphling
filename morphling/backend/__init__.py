"""Backend module for distributed matrix multiplication via various protocols."""

from morphling._Msg import (
    ProxyCli,
    ProxySvr,
)

from .base import BaseBackend, MatMulRequestMessage, MatMulResponseMessage
from .rabbitmq import RabbitMQBackend, RabbitMQWorker


# auto backend from name
class AutoBackend:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ backend")
            return RabbitMQBackend(*args, **kwargs)
        elif name == "proxy":
            print("Using Proxy backend")
            return ProxySvr()
        else:
            raise ValueError(f"Unknown backend: {name}")


class AutoWorker:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ worker")
            return RabbitMQWorker(*args, **kwargs)
        elif name == "proxy":
            print("Using Proxy worker")
            return ProxyCli()
        else:
            raise ValueError(f"Unknown worker: {name}")


__all__ = [
    "ProxyCli",
    "ProxySvr",
    "BaseBackend",
    "MatMulRequestMessage",
    "MatMulResponseMessage",
    "RabbitMQBackend",
    "RabbitMQWorker",
    "AutoBackend",
    "AutoWorker",
]
