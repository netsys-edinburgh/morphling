from .base import BaseBackend
from .rabbitmq import RabbitMQBackend, RabbitMQWorker


# auto backend from name
class AutoBackend:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ backend")
            return RabbitMQBackend(*args, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {name}")


class AutoWorker:
    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name == "rabbitmq":
            print("Using RabbitMQ worker")
            return RabbitMQWorker(*args, **kwargs)
        else:
            raise ValueError(f"Unknown worker: {name}")
