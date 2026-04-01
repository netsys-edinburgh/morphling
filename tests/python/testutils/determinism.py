from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np
import pytest
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    manual_seed = getattr(torch, "manual_seed")  # pyright: ignore[reportAny]
    _ = manual_seed(seed)  # pyright: ignore[reportAny]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def deterministic_context():
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    has_det_algorithms = hasattr(torch, "are_deterministic_algorithms_enabled")
    prev_det_algorithms = (
        torch.are_deterministic_algorithms_enabled()
        if has_det_algorithms
        else None
    )
    prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass
        yield
    finally:
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if prev_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas
        if prev_det_algorithms is not None:
            try:
                torch.use_deterministic_algorithms(prev_det_algorithms)
            except RuntimeError:
                pass


@pytest.fixture(scope="function", autouse=False)
def seed_fixture():
    set_all_seeds(42)
    with deterministic_context():
        yield
