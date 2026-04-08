from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory
from typing import Optional, cast

import numpy as np
import torch

from morphling.emulation.barrier import StructuralBarrier


class ShmAllReduceCommFn:
    def __init__(
        self,
        rank: int,
        world_size: int,
        tensor_size: int,
        barrier: StructuralBarrier,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._rank: int = rank
        self._world_size: int = world_size
        self._tensor_size: int = tensor_size
        self._barrier: StructuralBarrier = barrier
        self._dtype: torch.dtype = dtype
        self._step_idx: int = 0
        self._last_result: Optional[torch.Tensor] = None

    @property
    def last_result(self) -> Optional[torch.Tensor]:
        return self._last_result

    def __call__(self) -> None:
        step = self._step_idx
        grad_name = f"emulation_grad_{step}_{self._rank}"
        result_name = f"emulation_result_{step}"

        np_dtype = self._numpy_dtype()
        grad_np = np.random.rand(self._tensor_size).astype(np_dtype, copy=False)
        shm_grad = self._create_segment(grad_name, grad_np.nbytes)
        grad_buf = cast(memoryview, shm_grad.buf)
        np.ndarray((self._tensor_size,), dtype=np_dtype, buffer=grad_buf)[:] = (
            grad_np
        )

        _ = self._barrier.barrier(self._rank, "comm_write", step, 0.0)

        if self._rank == 0:
            total = np.zeros(self._tensor_size, dtype=np_dtype)
            for r in range(self._world_size):
                src = SharedMemory(name=f"emulation_grad_{step}_{r}")
                try:
                    src_buf = cast(memoryview, src.buf)
                    src_np = np.ndarray(
                        (self._tensor_size,), dtype=np_dtype, buffer=src_buf
                    ).copy()
                    total += src_np
                finally:
                    src.close()
            avg = total / float(self._world_size)
            shm_result = self._create_segment(result_name, avg.nbytes)
            result_buf = cast(memoryview, shm_result.buf)
            np.ndarray((self._tensor_size,), dtype=np_dtype, buffer=result_buf)[
                :
            ] = avg
            shm_result.close()

        _ = self._barrier.barrier(self._rank, "comm_read", step, 0.0)

        shm_result_read = SharedMemory(name=result_name)
        try:
            out_buf = cast(memoryview, shm_result_read.buf)
            out = np.ndarray(
                (self._tensor_size,), dtype=np_dtype, buffer=out_buf
            ).copy()
        finally:
            shm_result_read.close()
        self._last_result = torch.from_numpy(out)

        _ = self._barrier.barrier(self._rank, "comm_cleanup", step, 0.0)

        shm_grad.close()
        self._unlink_if_exists(grad_name)
        if self._rank == 0:
            self._unlink_if_exists(result_name)

        self._step_idx += 1

    @staticmethod
    def _create_segment(name: str, size: int) -> SharedMemory:
        try:
            return SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            stale = SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            return SharedMemory(name=name, create=True, size=size)

    @staticmethod
    def _unlink_if_exists(name: str) -> None:
        try:
            shm = SharedMemory(name=name, create=False)
        except FileNotFoundError:
            return
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
        finally:
            shm.close()

    def _numpy_dtype(self) -> np.dtype[np.float32] | np.dtype[np.float64]:
        if self._dtype == torch.float32:
            return np.dtype(np.float32)
        if self._dtype == torch.float64:
            return np.dtype(np.float64)
        raise ValueError(f"Unsupported dtype for SHM allreduce: {self._dtype}")
