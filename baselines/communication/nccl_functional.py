"""Functional NCCL send/recv/allreduce with CuPy-optional fallback.

When CuPy NCCL is available, uses direct CuPy NCCL communicator calls.
When unavailable, falls back to torch.distributed send/recv/all_reduce.
"""
from __future__ import annotations

import importlib
import logging
from collections import deque
from typing import Any, cast

logger = logging.getLogger(__name__)

# ── Async-send bookkeeping (avoids send-send deadlocks) ──
# Each entry is (Work, cpu_tensor) so the CPU buffer stays
# alive until the non-blocking isend completes.
_pending_send_ops: deque[tuple[Any, Any]] = deque()


def _cupy_module_name() -> str:
    return "".join(("cu", "py"))


def _cupy_nccl_module_name() -> str:
    return ".".join((_cupy_module_name(), "cuda", "nccl"))

_has_cupy = False
try:
    importlib.import_module(_cupy_module_name())
    importlib.import_module(_cupy_nccl_module_name())

    _has_cupy = True
except ImportError:
    pass

_logged_path = False


def _log_path_once(using_cupy: bool) -> None:
    """Log which comm path is active, only once."""
    global _logged_path
    if not _logged_path:
        path = "CuPy NCCL" if using_cupy else "torch.distributed"
        logger.info("nccl_functional: using %s path", path)
        _logged_path = True


def _import_cupy() -> Any:
    return importlib.import_module(_cupy_module_name())


def _import_cupy_nccl() -> Any:
    return importlib.import_module(_cupy_nccl_module_name())


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _import_dist() -> Any:
    return importlib.import_module("torch.distributed")


def _type_torch_to_nccl(torch_dtype: object) -> int:
    """Map torch dtype → CuPy NCCL dtype constant."""
    nccl = _import_cupy_nccl()
    torch = _import_torch()
    return {
        torch.float32: nccl.NCCL_FLOAT32,
        torch.float: nccl.NCCL_FLOAT32,
        torch.float16: nccl.NCCL_FLOAT16,
        torch.float64: nccl.NCCL_FLOAT64,
        torch.int32: nccl.NCCL_INT32,
        torch.int: nccl.NCCL_INT,
        torch.uint8: nccl.NCCL_UINT8,
    }[torch_dtype]


def _nccl_send(
    tensor: object,
    dst_rank: int,
    nccl_comm: object,
    stream: object | None = None,
) -> None:
    """P2P send via CuPy NCCL communicator."""
    cupy = _import_cupy()
    _log_path_once(True)
    t = cast(Any, tensor)
    assert t.is_contiguous() and t.is_cuda
    torch = _import_torch()
    if stream is None:
        stream = torch.cuda.current_stream(t.device)
    s = cast(Any, stream)
    cupy_stream = cupy.cuda.ExternalStream(s.cuda_stream)
    comm = cast(Any, nccl_comm)
    comm.send(
        t.data_ptr(),
        t.numel(),
        _type_torch_to_nccl(t.dtype),
        dst_rank,
        cupy_stream.ptr,
    )


def _nccl_recv(
    tensor: object,
    src_rank: int,
    nccl_comm: object,
    stream: object | None = None,
) -> None:
    """P2P recv via CuPy NCCL communicator."""
    cupy = _import_cupy()
    _log_path_once(True)
    t = cast(Any, tensor)
    assert t.is_contiguous() and t.is_cuda
    torch = _import_torch()
    if stream is None:
        stream = torch.cuda.current_stream(t.device)
    s = cast(Any, stream)
    cupy_stream = cupy.cuda.ExternalStream(s.cuda_stream)
    comm = cast(Any, nccl_comm)
    comm.recv(
        t.data_ptr(),
        t.numel(),
        _type_torch_to_nccl(t.dtype),
        src_rank,
        cupy_stream.ptr,
    )


def _nccl_allreduce(
    tensor: object,
    nccl_comm: object,
    stream: object | None = None,
) -> None:
    """In-place AllReduce sum via CuPy NCCL."""
    cupy = _import_cupy()
    nccl = _import_cupy_nccl()
    _log_path_once(True)
    t = cast(Any, tensor)
    assert t.is_contiguous() and t.is_cuda
    torch = _import_torch()
    if stream is None:
        stream = torch.cuda.current_stream(t.device)
    s = cast(Any, stream)
    cupy_stream = cupy.cuda.ExternalStream(s.cuda_stream)
    comm = cast(Any, nccl_comm)
    comm.allReduce(
        t.data_ptr(),
        t.data_ptr(),
        t.numel(),
        _type_torch_to_nccl(t.dtype),
        nccl.NCCL_SUM,
        cupy_stream.ptr,
    )


def _dist_send(
    tensor: object,
    dst_rank: int,
    stream: object | None = None,
) -> None:
    """P2P send via torch.distributed (fallback).

    Uses non-blocking ``dist.isend`` to prevent send-send
    deadlocks in 1F1B pipeline schedules where adjacent
    stages simultaneously send to each other.
    """
    _log_path_once(False)
    dist = _import_dist()
    t = cast(Any, tensor)
    # Gloo cannot send GPU tensors; move to CPU
    if t.is_cuda:
        cpu_t = t.data.cpu()
    else:
        cpu_t = t.clone()
    # Non-blocking send — lets the caller proceed to recv
    work = dist.isend(cpu_t, dst=dst_rank)
    _pending_send_ops.append((work, cpu_t))
    # Eagerly drain completed ops to bound memory
    _drain_completed_sends()


def _drain_completed_sends() -> None:
    """Remove completed async sends to free CPU buffers."""
    while _pending_send_ops:
        work, _ = _pending_send_ops[0]
        if work.is_completed():
            _pending_send_ops.popleft()
        else:
            break


def flush_all_sends() -> None:
    """Block until every pending async send completes."""
    while _pending_send_ops:
        work, _ = _pending_send_ops.popleft()
        work.wait()


def _dist_recv(
    tensor: object,
    src_rank: int,
    stream: object | None = None,
) -> None:
    """P2P recv via torch.distributed (fallback)."""
    _log_path_once(False)
    dist = _import_dist()
    torch = _import_torch()
    t = cast(Any, tensor)
    orig_device = t.device
    # Gloo cannot recv into GPU tensors; use CPU buffer
    if t.is_cuda:
        cpu_buf = t.data.cpu()
        if stream is not None:
            with torch.cuda.stream(cast(Any, stream)):
                dist.recv(cpu_buf, src=src_rank)
        else:
            dist.recv(cpu_buf, src=src_rank)
        t.data.copy_(cpu_buf)
    else:
        if stream is not None:
            with torch.cuda.stream(cast(Any, stream)):
                dist.recv(t, src=src_rank)
        else:
            dist.recv(t, src=src_rank)


def _dist_allreduce(
    tensor: object,
    stream: object | None = None,
) -> None:
    """In-place AllReduce via torch.distributed (fallback)."""
    _log_path_once(False)
    dist = _import_dist()
    torch = _import_torch()
    t = cast(Any, tensor)
    # Gloo cannot allreduce GPU tensors; use CPU buffer
    if t.is_cuda:
        cpu_buf = t.data.cpu()
        if stream is not None:
            with torch.cuda.stream(cast(Any, stream)):
                dist.all_reduce(cpu_buf)
        else:
            dist.all_reduce(cpu_buf)
        t.data.copy_(cpu_buf)
    else:
        if stream is not None:
            with torch.cuda.stream(cast(Any, stream)):
                dist.all_reduce(t)
        else:
            dist.all_reduce(t)


def functional_send(
    tensor: object,
    dst_rank: int,
    nccl_comm: object | None = None,
    stream: object | None = None,
) -> None:
    """Send tensor to dst_rank. Uses CuPy NCCL if available."""
    if _has_cupy and nccl_comm is not None:
        _nccl_send(tensor, dst_rank, nccl_comm, stream)
    else:
        _dist_send(tensor, dst_rank, stream)


def functional_recv(
    tensor: object,
    src_rank: int,
    nccl_comm: object | None = None,
    stream: object | None = None,
) -> None:
    """Recv tensor from src_rank. Uses CuPy NCCL if available."""
    if _has_cupy and nccl_comm is not None:
        _nccl_recv(tensor, src_rank, nccl_comm, stream)
    else:
        _dist_recv(tensor, src_rank, stream)


def functional_allreduce(
    tensor: object,
    nccl_comm: object | None = None,
    stream: object | None = None,
) -> None:
    """AllReduce tensor. Uses CuPy NCCL if available."""
    if _has_cupy and nccl_comm is not None:
        _nccl_allreduce(tensor, nccl_comm, stream)
    else:
        _dist_allreduce(tensor, stream)


def has_cupy_nccl() -> bool:
    """Return whether CuPy NCCL is available."""
    return _has_cupy


__all__ = [
    "functional_send",
    "functional_recv",
    "functional_allreduce",
    "flush_all_sends",
    "has_cupy_nccl",
]
