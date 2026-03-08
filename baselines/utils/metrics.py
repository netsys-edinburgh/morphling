"""JSONL metrics collector for experiment instrumentation.

Each rank writes raw timestamps and byte counts to a local
JSONL file.  Derived metrics (durations, aggregations) are
computed post-training by the stitcher.

Usage
-----
    from baselines.utils.metrics import MetricsCollector

    mc = MetricsCollector(rank=rank, out_dir="/tmp/asteroid_metrics")
    mc.log_load(model_load_start_ts=..., model_load_end_ts=..., ...)

    for iter_num in range(...):
        mc.iter_start(iter_num)          # records start_ts
        mc.fwd_start()                   # records forward_start_ts
        ...
        mc.fwd_end()                     # records forward_end_ts
        ...
        mc.iter_end(loss=..., lr=..., memory_allocated_mb=...)

    mc.close()
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    """ISO-8601 UTC timestamp with microseconds."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
    )


def _now_mono() -> float:
    """Monotonic clock (seconds) for pairing start/end."""
    return time.monotonic()


class MetricsCollector:
    """Append-only JSONL writer for per-rank metrics.

    Thread-safety: NOT thread-safe.  Must be called from the
    rank's main training thread only.
    """

    # Expose as static so callers can timestamp without
    # an instance (e.g. before __init__).
    _now_iso = staticmethod(_now_iso)

    def __init__(
        self,
        rank: int,
        out_dir: str = "/tmp/asteroid_metrics",
        enabled: bool = True,
    ) -> None:
        self._rank = rank
        self._enabled = enabled
        self._fp = None

        # Per-iter scratch (reset each iter_start)
        # Initialized unconditionally so attributes exist
        # even when disabled (avoids AttributeError).
        self._iter: int = -1
        self._iter_start_ts: str = ""
        self._fwd_start_ts: str = ""
        self._fwd_end_ts: str = ""
        self._bwd_start_ts: str = ""
        self._bwd_end_ts: str = ""
        self._opt_start_ts: str = ""
        self._opt_end_ts: str = ""
        self._pp_send_start_ts: str = ""
        self._pp_send_end_ts: str = ""
        self._pp_recv_start_ts: str = ""
        self._pp_recv_end_ts: str = ""
        self._dp_ar_start_ts: str = ""
        self._dp_ar_end_ts: str = ""
        self._pp_send_bytes: int = 0
        self._pp_recv_bytes: int = 0
        self._pp_send_count: int = 0
        self._pp_recv_count: int = 0
        self._dp_ar_bytes: int = 0

        if not enabled:
            return

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rank_{rank}.jsonl")
        self._fp = open(path, "w")  # noqa: SIM115 — "w" so each run starts clean

    # ── helpers ──────────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        if self._fp is not None:
            self._fp.write(json.dumps(record) + "\n")
            self._fp.flush()

    # ── load / init metrics ──────────────────────────────

    def log_load(
        self,
        model_load_start_ts: str | None = None,
        model_load_end_ts: str | None = None,
        nccl_init_start_ts: str | None = None,
        nccl_init_end_ts: str | None = None,
        memory_after_load_mb: float | None = None,
    ) -> None:
        """Emit a single 'load' record for this rank."""
        if not self._enabled:
            return
        self._write({
            "type": "load",
            "rank": self._rank,
            "model_load_start_ts": model_load_start_ts,
            "model_load_end_ts": model_load_end_ts,
            "nccl_init_start_ts": nccl_init_start_ts,
            "nccl_init_end_ts": nccl_init_end_ts,
            "memory_after_load_mb": memory_after_load_mb,
        })

    # ── per-iteration step markers ───────────────────────

    def iter_start(self, iter_num: int) -> None:
        if not self._enabled:
            return
        self._iter = iter_num
        self._iter_start_ts = _now_iso()
        # Reset per-iter accumulators
        self._fwd_start_ts = ""
        self._fwd_end_ts = ""
        self._bwd_start_ts = ""
        self._bwd_end_ts = ""
        self._opt_start_ts = ""
        self._opt_end_ts = ""
        self._pp_send_start_ts = ""
        self._pp_send_end_ts = ""
        self._pp_recv_start_ts = ""
        self._pp_recv_end_ts = ""
        self._dp_ar_start_ts = ""
        self._dp_ar_end_ts = ""
        self._pp_send_bytes = 0
        self._pp_recv_bytes = 0
        self._pp_send_count = 0
        self._pp_recv_count = 0
        self._dp_ar_bytes = 0

    def fwd_start(self) -> None:
        if not self._enabled:
            return
        self._fwd_start_ts = _now_iso()

    def fwd_end(self) -> None:
        if not self._enabled:
            return
        self._fwd_end_ts = _now_iso()

    def bwd_start(self) -> None:
        if not self._enabled:
            return
        self._bwd_start_ts = _now_iso()

    def bwd_end(self) -> None:
        if not self._enabled:
            return
        self._bwd_end_ts = _now_iso()

    def opt_start(self) -> None:
        if not self._enabled:
            return
        self._opt_start_ts = _now_iso()

    def opt_end(self) -> None:
        if not self._enabled:
            return
        self._opt_end_ts = _now_iso()

    def pp_send_start(self) -> None:
        if not self._enabled:
            return
        if not self._pp_send_start_ts:
            self._pp_send_start_ts = _now_iso()

    def pp_send_end(self, nbytes: int = 0) -> None:
        if not self._enabled:
            return
        self._pp_send_end_ts = _now_iso()
        self._pp_send_bytes += nbytes
        self._pp_send_count += 1

    def pp_recv_start(self) -> None:
        if not self._enabled:
            return
        if not self._pp_recv_start_ts:
            self._pp_recv_start_ts = _now_iso()

    def pp_recv_end(self, nbytes: int = 0) -> None:
        if not self._enabled:
            return
        self._pp_recv_end_ts = _now_iso()
        self._pp_recv_bytes += nbytes
        self._pp_recv_count += 1

    def dp_allreduce_start(self) -> None:
        if not self._enabled:
            return
        self._dp_ar_start_ts = _now_iso()

    def dp_allreduce_end(self, nbytes: int = 0) -> None:
        if not self._enabled:
            return
        self._dp_ar_end_ts = _now_iso()
        self._dp_ar_bytes = nbytes

    def iter_end(
        self,
        loss: float | None = None,
        lr: float | None = None,
        memory_allocated_mb: float | None = None,
        memory_reserved_mb: float | None = None,
        memory_peak_mb: float | None = None,
    ) -> None:
        """Emit the complete 'step' record for this iteration."""
        if not self._enabled:
            return
        end_ts = _now_iso()
        self._write({
            "type": "step",
            "rank": self._rank,
            "iter": self._iter,
            "start_ts": self._iter_start_ts,
            "end_ts": end_ts,
            "forward_start_ts": self._fwd_start_ts or None,
            "forward_end_ts": self._fwd_end_ts or None,
            "backward_start_ts": self._bwd_start_ts or None,
            "backward_end_ts": self._bwd_end_ts or None,
            "optimizer_start_ts": self._opt_start_ts or None,
            "optimizer_end_ts": self._opt_end_ts or None,
            "pp_send_start_ts": (
                self._pp_send_start_ts or None
            ),
            "pp_send_end_ts": (
                self._pp_send_end_ts or None
            ),
            "pp_recv_start_ts": (
                self._pp_recv_start_ts or None
            ),
            "pp_recv_end_ts": (
                self._pp_recv_end_ts or None
            ),
            "pp_send_count": self._pp_send_count,
            "pp_recv_count": self._pp_recv_count,
            "pp_send_bytes": self._pp_send_bytes,
            "pp_recv_bytes": self._pp_recv_bytes,
            "dp_allreduce_start_ts": (
                self._dp_ar_start_ts or None
            ),
            "dp_allreduce_end_ts": (
                self._dp_ar_end_ts or None
            ),
            "dp_allreduce_bytes": self._dp_ar_bytes,
            "loss": loss,
            "lr": lr,
            "memory_allocated_mb": memory_allocated_mb,
            "memory_reserved_mb": memory_reserved_mb,
            "memory_peak_mb": memory_peak_mb,
        })

    # ── global step raw (rank 0 only) ────────────────────

    def log_global_step_raw(
        self,
        iter_num: int,
        loss: float | None,
        lr: float | None,
        iter_start_ts: str,
        iter_end_ts: str,
    ) -> None:
        """Rank 0 logs raw global-level values per iter."""
        if not self._enabled:
            return
        self._write({
            "type": "global_step_raw",
            "iter": iter_num,
            "loss": loss,
            "lr": lr,
            "iter_start_ts": iter_start_ts,
            "iter_end_ts": iter_end_ts,
        })

    # ── lifecycle ────────────────────────────────────────

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
