import logging
from dataclasses import dataclass, field
from typing import Union

import psutil
import torch
from transformers import HfArgumentParser


@dataclass
class DeviceConfig:
    pass


@dataclass
class EmulatorConfig:
    gpu_memory: Union[int, float] = field(
        default=0.9, metadata={"help": "Can be a float in the range [0, 1] or an integer in GB"}
    )
    cpu_memory: Union[int, float] = field(
        default=0.5, metadata={"help": "Can be a float in the range [0, 1] or an integer in GB"}
    )
    ckpt_path: str = field(
        default="checkpoints", metadata={"help": "The path to save the emulator model checkpoints"}
    )

    def __post_init__(self):
        if self.gpu_memory < 0 or self.gpu_memory > 1:
            raise ValueError("gpu_memory should be a float in the range [0, 1]")
        if self.cpu_memory < 0 or self.cpu_memory > 1:
            raise ValueError("cpu_memory should be a float in the range [0, 1]")

        total_gpu_memory = int(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)
        total_cpu_memory = int(psutil.virtual_memory().total / 1024 ** 3)

        if isinstance(self.gpu_memory, float):
            self.gpu_memory = int(total_gpu_memory * self.gpu_memory)

        if isinstance(self.cpu_memory, float):
            self.cpu_memory = int(total_cpu_memory * self.cpu_memory)

        if self.gpu_memory > total_gpu_memory:
            raise ValueError(f"gpu_memory should not exceed {total_gpu_memory} GB")
        if self.cpu_memory > total_cpu_memory:
            raise ValueError(f"cpu_memory should not exceed {total_cpu_memory} GB")


    @classmethod
    def load_from_file(self, config_path):
        parser = HfArgumentParser(self)
        self = parser.parse_json_file(json_file=config_path)[0]
        return self

    @classmethod
    def load_from_json(self, config_json):
        parser = HfArgumentParser(self)
        self = parser.parse_dict(config_json)[0]
        return self


# a universal logger for all modules
# Usage:
# from morphling.common.logger import logger
# logger.info("hello world")
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, LOG_LEVEL.upper())

from logging import Formatter, StreamHandler, getLogger


def get_logger():
    logger = getLogger("morphling")
    logger.setLevel(LOG_LEVEL)
    handler = StreamHandler()
    handler.setLevel(LOG_LEVEL)
    formatter = Formatter("%(asctime)s - %(className)s.%(funcName)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger




