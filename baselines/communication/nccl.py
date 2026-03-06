from __future__ import annotations

import importlib
import logging
from typing import Any, cast

from .nccl_functional import (
    _PendingRecv,
    flush_all_sends,
    functional_allreduce,
    functional_recv,
    functional_recv_async,
    functional_send,
    has_cupy_nccl,
)

logger = logging.getLogger(__name__)


def _import_cupy() -> Any:
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise RuntimeError("CuPy is required for NCCLBackend.") from exc

def _import_nccl() -> Any:
    try:
        return importlib.import_module("cupy.cuda.nccl")
    except ImportError as exc:
        raise RuntimeError("CuPy NCCL is required.") from exc

def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("torch is required for NCCLBackend.") from exc


def _import_dist() -> Any:
    try:
        return importlib.import_module("torch.distributed")
    except ImportError as exc:
        raise RuntimeError("torch.distributed is required.") from exc


def _uid_to_bytes(uid: object) -> bytes:
    """Serialize NCCL unique ID for store exchange.

    CuPy 13.x returns a tuple of signed ints;
    CuPy 14+ may return bytes directly.
    """
    if isinstance(uid, bytes):
        return uid
    if isinstance(uid, (tuple, list)):
        import numpy as np

        return np.array(uid, dtype=np.int8).tobytes()
    return bytes(cast(Any, uid))


def _bytes_to_uid(data: bytes) -> object:
    """Deserialize NCCL unique ID from store.

    Returns tuple (CuPy 13.x) or bytes (CuPy 14+)
    based on what NcclCommunicator expects.
    """
    import numpy as np

    # CuPy 13.x NcclCommunicator expects a tuple
    arr = np.frombuffer(data, dtype=np.int8)
    return tuple(int(x) for x in arr)

def _type_torch_to_nccl(torch_dtype: object) -> int:
    torch = _import_torch()
    nccl = _import_nccl()
    dtype_map = {
        torch.float32: nccl.NCCL_FLOAT32,
        torch.float: nccl.NCCL_FLOAT32,
        torch.float16: nccl.NCCL_FLOAT16,
        torch.bfloat16: nccl.NCCL_BFLOAT16,
        torch.float64: nccl.NCCL_FLOAT64,
        torch.int32: nccl.NCCL_INT32,
        torch.int: nccl.NCCL_INT,
        torch.uint8: nccl.NCCL_UINT8,
    }
    try:
        return dtype_map[torch_dtype]
    except KeyError as exc:
        msg = f"Unsupported dtype for NCCL send/recv: {torch_dtype}"
        raise TypeError(msg) from exc


class NCCLBackend:
    """CuPy NCCL P2P send/recv + AllReduce wrapper."""

    rank: int
    world_size: int
    cuda_id: int
    pp_nccl_comm: object | None
    dp_nccl_comm: object | None

    def __init__(self, rank: int, world_size: int, cuda_id: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.cuda_id = cuda_id
        self.pp_nccl_comm = None
        self.dp_nccl_comm = None

    def _barrier(self) -> None:
        """torch.distributed barrier.

        Uses the existing process-group NCCL communicator
        (created lazily by torch on the first collective).
        On a fresh deploy all ranks participate and it
        succeeds immediately.  On a stale restart the
        missing peers cause a clean 300 s timeout instead
        of silently passing with stale store keys.
        """
        dist = _import_dist()
        if not (dist.is_available() and dist.is_initialized()):
            return
        dist.barrier()

    def _resolve_stream(
        self,
        tensor: object,
        stream: object | None,
    ) -> object:
        if stream is not None:
            return stream

        tensor_any = cast(Any, tensor)
        torch = _import_torch()
        if tensor_any.is_cuda:
            return torch.cuda.current_stream(tensor_any.device)
        return torch.cuda.current_stream(self.cuda_id)

    def send(
        self,
        tensor: object,
        dst_rank: int,
        stream: object | None = None,
    ) -> None:
        if has_cupy_nccl() and self.pp_nccl_comm is None:
            raise RuntimeError("PP communicator is not initialized.")

        tensor_any = cast(Any, tensor)
        if not tensor_any.is_cuda:
            raise ValueError("NCCL send expects a CUDA tensor.")
        if not tensor_any.is_contiguous():
            tensor_any = tensor_any.contiguous()

        stream_resolved = self._resolve_stream(tensor_any, stream)
        functional_send(
            tensor_any,
            dst_rank,
            self.pp_nccl_comm,
            stream_resolved,
        )

    def recv(
        self,
        tensor: object,
        src_rank: int,
        stream: object | None = None,
    ) -> None:
        if has_cupy_nccl() and self.pp_nccl_comm is None:
            raise RuntimeError("PP communicator is not initialized.")

        tensor_any = cast(Any, tensor)
        if not tensor_any.is_cuda:
            raise ValueError("NCCL recv expects a CUDA tensor.")
        if not tensor_any.is_contiguous():
            tensor_any = tensor_any.contiguous()

        stream_resolved = self._resolve_stream(tensor_any, stream)
        functional_recv(
            tensor_any,
            src_rank,
            self.pp_nccl_comm,
            stream_resolved,
        )

    def recv_async(
        self,
        tensor: object,
        src_rank: int,
        stream: object | None = None,
    ) -> _PendingRecv | None:
        """Post a non-blocking recv and return a handle.

        Call ``handle.wait()`` before consuming the data.
        Returns ``None`` when CuPy NCCL handles the recv
        on the CUDA stream (no CPU-side wait needed).
        """
        if has_cupy_nccl() and self.pp_nccl_comm is None:
            raise RuntimeError(
                "PP communicator is not initialized."
            )

        tensor_any = cast(Any, tensor)
        if not tensor_any.is_cuda:
            raise ValueError(
                "NCCL recv expects a CUDA tensor."
            )
        if not tensor_any.is_contiguous():
            tensor_any = tensor_any.contiguous()

        stream_resolved = self._resolve_stream(
            tensor_any, stream
        )
        return functional_recv_async(
            tensor_any,
            src_rank,
            self.pp_nccl_comm,
            stream_resolved,
        )

    def allreduce(
        self,
        tensor: object,
        stream: object | None = None,
    ) -> None:
        nccl_comm = self.dp_nccl_comm or self.pp_nccl_comm
        if has_cupy_nccl() and nccl_comm is None:
            raise RuntimeError("No NCCL communicator is initialized.")

        tensor_any = cast(Any, tensor)
        if not tensor_any.is_cuda:
            raise ValueError("NCCL allreduce expects a CUDA tensor.")
        if not tensor_any.is_contiguous():
            tensor_any = tensor_any.contiguous()

        stream_resolved = self._resolve_stream(tensor_any, stream)
        functional_allreduce(
            tensor_any,
            nccl_comm,
            stream_resolved,
        )

    def flush_sends(self) -> None:
        """Wait for all pending non-blocking sends to finish."""
        flush_all_sends()

    def setup_communicators(
        self,
        pp_rank: int,
        dp_rank: int,
        pp_size: int,
        dp_size: int,
        dist_store: object | None,
        pp_size_override: int | None = None,
        dp_size_override: int | None = None,
    ) -> tuple[object, object | None]:
        dist = _import_dist()
        effective_pp_size = (
            pp_size_override
            if pp_size_override is not None
            else pp_size
        )
        effective_dp_size = (
            dp_size_override
            if dp_size_override is not None
            else dp_size
        )

        if not has_cupy_nccl():
            self.pp_nccl_comm = None
            self.dp_nccl_comm = None
            logger.warning(
                "CuPy NCCL unavailable; using torch.distributed "
                "fallback path.",
            )
            return self.pp_nccl_comm, self.dp_nccl_comm

        cupy = _import_cupy()
        nccl = _import_nccl()
        cupy.cuda.Device(self.cuda_id).use()

        store = dist_store
        if store is None:
            store = dist.distributed_c10d._get_default_store()
        store_any = cast(Any, store)

        pp_comm_name = f"baseline_pp_{dp_rank}"
        pp_uid_raw = nccl.get_unique_id()
        if pp_rank == 0:
            uid_bytes = _uid_to_bytes(pp_uid_raw)
            store_any.set(
                f"group-{pp_comm_name}-uid",
                uid_bytes,
            )
        self._barrier()
        if pp_rank != 0:
            uid_data = store_any.get(f"group-{pp_comm_name}-uid")
            raw = uid_data if isinstance(uid_data, bytes) else bytes(uid_data)
            pp_uid_raw = _bytes_to_uid(raw)
        self.pp_nccl_comm = nccl.NcclCommunicator(
            effective_pp_size,
            pp_uid_raw,
            pp_rank,
        )

        self.dp_nccl_comm = None
        if effective_dp_size > 1:
            dp_comm_name = f"baseline_dp_{pp_rank}"
            dp_uid_raw = nccl.get_unique_id()
            if dp_rank == 0:
                uid_dp_bytes = _uid_to_bytes(dp_uid_raw)
                store_any.set(
                    f"group-{dp_comm_name}-uid",
                    uid_dp_bytes,
                )
            self._barrier()
            if dp_rank != 0:
                uid_dp_data = store_any.get(f"group-{dp_comm_name}-uid")
                raw_dp = (
                    uid_dp_data
                    if isinstance(uid_dp_data, bytes)
                    else bytes(uid_dp_data)
                )
                dp_uid_raw = _bytes_to_uid(raw_dp)
            self.dp_nccl_comm = nccl.NcclCommunicator(
                effective_dp_size,
                dp_uid_raw,
                dp_rank,
            )

        logger.debug(
            "NCCL communicators initialized: rank=%s pp=%s dp=%s",
            self.rank,
            effective_pp_size,
            effective_dp_size,
        )
        return self.pp_nccl_comm, self.dp_nccl_comm


__all__ = ["NCCLBackend", "_type_torch_to_nccl"]
