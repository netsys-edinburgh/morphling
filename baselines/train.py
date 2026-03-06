"""Distributed training entry point for baselines framework.

Supports two launch modes:
  1. mp.spawn (single-node):
       python -m baselines.train --spawn --num-gpus 4 \\
           --config baselines/configs/dtfm_default.yaml
  2. Per-process (torchrun / SSH / manual):
       torchrun --nproc_per_node=4 -m baselines.train \\
           --config baselines/configs/dtfm_default.yaml \\
           --world-size 4

Based on DT-FM launch pattern:
  https://github.com/xyf2002/DT-FM
"""
from __future__ import annotations

import argparse
import json
import logging
import fcntl
import os
import sys
import time
import threading
import traceback
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from baselines.communication.nccl import NCCLBackend
from baselines.communication.nccl_functional import (
    has_cupy_nccl,
    nccl_group_start,
    nccl_group_end,
)
from baselines.core.config import (
    BaseConfig,
    DeviceConfig,
    DeviceTopology,
    GreenCtxConfig,
    ParallelismPlan,
)
from baselines.models import PipelineStage
from baselines.strategies import (
    AsteroidStrategy,
    ConfidentStrategy,
    DTFMStrategy,
)
from baselines.utils.config_loader import load_config
from baselines.utils.flatten import flatten_params
from baselines.utils.lr_schedule import get_lr
from baselines.utils.seed import seed_everything
from baselines.core.pipeline_schedule import build_schedule
from baselines.utils.mps import (
    get_mps_client_env,
    start_mps,
    stop_all_mps,
)

logger = logging.getLogger(__name__)


# ── Synthetic data loader ────────────────────────────────


class SyntheticDataLoader:
    """Wraps pre-generated tensors into micro-batch iteration."""

    def __init__(
        self,
        embeds: torch.Tensor,
        labels: torch.Tensor,
        micro_batch_size: int,
    ) -> None:
        self.embeds = embeds
        self.labels = labels
        self.micro_batch_size = micro_batch_size
        self.idx = 0

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        end = self.idx + self.micro_batch_size
        if end > len(self.embeds):
            self.idx = 0
            end = self.micro_batch_size
        x = self.embeds[self.idx : end]
        y = self.labels[self.idx : end]
        self.idx = end
        return x, y


# ── Worker ────────────────────────────────────────────────

# Watchdog: updated by the training loop to prove liveness.
# A daemon thread calls os._exit(1) after _WD_TIMEOUT
# seconds of no progress so peers that lost a rank don't
# hang forever in a blocking NCCL recv.
_wd_ts: float = 0.0
_WD_TIMEOUT = 300          # 5 min — longer than any iter


def _watchdog_loop(rank: int) -> None:
    """Daemon thread: exit the process when stuck."""
    global _wd_ts
    while True:
        time.sleep(15)
        if _wd_ts > 0 and time.time() - _wd_ts > _WD_TIMEOUT:
            print(
                f"[RANK {rank}] WATCHDOG: no progress for "
                f"{_WD_TIMEOUT}s — peer may be dead. "
                f"Exiting.",
                flush=True,
            )
            os._exit(1)


def worker(
    rank: int,
    cfg: BaseConfig,
    plan: ParallelismPlan,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """Per-rank training worker (DT-FM pattern)."""
    try:
        _worker_impl(rank, cfg, plan, train_data, val_data, args)
    except Exception as exc:
        print(
            f"[RANK {rank}] FATAL: {exc}",
            flush=True,
        )
        traceback.print_exc()
        # Hard exit so K8s restarts this rank immediately
        # instead of waiting for Python cleanup.
        os._exit(1)


def _worker_impl(
    rank: int,
    cfg: BaseConfig,
    plan: ParallelismPlan,
    train_data: tuple[torch.Tensor, torch.Tensor],
    val_data: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    pp_size = cfg.parallel.pp_size
    dp_size = cfg.parallel.dp_size
    world_size = cfg.distributed.world_size

    # ── Step 1: Rank geometry ────────────────────────────
    pp_rank = rank % pp_size
    dp_rank = rank // pp_size

    # ── Step 2: Seed by pp_rank for identical DP weights ─
    torch.manual_seed(cfg.training.seed + pp_rank)

    # ── Step 3: GPU assignment ───────────────────────────
    if args.cuda_id >= 0:
        # Explicit --cuda-id flag (K8s per-node scenario)
        cuda_id = args.cuda_id
    else:
        cuda_id = _resolve_cuda_id(rank, plan, pp_size)
    device = torch.device("cuda", cuda_id)

    # Inject MPS client env before any CUDA call.
    if cfg.device.mps_enabled:
        mps_env = get_mps_client_env(
            cuda_id, cfg.device
        )
        for _k, _v in mps_env.items():
            os.environ[_k] = _v

    torch.cuda.set_device(device)
    print(
        f"[RANK {rank}] pp_rank={pp_rank} dp_rank={dp_rank}"
        f" cuda:{cuda_id}",
        flush=True,
    )

    # ── Step 3b: Green context controller ───────────────
    gc_controller = None
    if cfg.greenctx.enabled:
        from morphling.runtime.green_context import (
            GreenContextConfig,
            GreenContextController,
        )
        gc_cfg = GreenContextConfig(
            enabled=True,
            backend=cfg.greenctx.backend,
            trace_path=cfg.greenctx.trace_path,
            clock_mode=cfg.greenctx.clock_mode,
            strict=cfg.greenctx.strict,
            switch_sync=cfg.greenctx.switch_sync,
            num_partitions=cfg.greenctx.num_partitions,
            partition_idx=cfg.greenctx.partition_idx,
            stream_priority=cfg.greenctx.stream_priority,
        )
        gc_controller = (
            GreenContextController.from_config(
                device_id=cuda_id, cfg=gc_cfg,
            )
        )
        print(
            f"[RANK {rank}] GreenCtx: "
            f"supported={gc_controller.is_supported}"
            f" sms={gc_controller.available_sm_counts()}",
            flush=True,
        )

    # ── Step 4: Stage boundaries ─────────────────────────
    is_first = pp_rank == 0
    is_last = pp_rank == pp_size - 1
    is_single = pp_size == 1
    num_stages = len(plan.partition_points) + 1
    boundaries = (
        [0]
        + list(plan.partition_points)
        + [cfg.model.num_layers]
    )
    if num_stages == 1:
        boundaries = [0, cfg.model.num_layers]

    # ── Step 5: Init process group ───────────────────────
    # Use Gloo for control-plane collectives (barrier,
    # new_group).  P2P and allreduce use CuPy NCCL
    # directly to avoid conflicts between torch.distributed
    # NCCL communicators and CuPy NCCL communicators.
    _comm_backend = getattr(cfg.parallel, "comm_backend", "gloo")
    if _comm_backend in ("torch_dist", "nccl"):
        _comm_backend = "gloo"
    # Validate backend availability
    if _comm_backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeError("Gloo backend not available")

    if not dist.is_initialized():
        # Detect K8s / multi-node: if MASTER_ADDR is set in
        # the environment, use env:// init (like asteroid_project
        # worker.py) so torch reads MASTER_ADDR, MASTER_PORT,
        # RANK, WORLD_SIZE from env vars injected by the
        # K8s manifests.
        use_env_init = "MASTER_ADDR" in os.environ
        if use_env_init:
            init_method = "env://"
            init_timeout = 300
            print(
                f"[RANK {rank}] init_process_group"
                f"({_comm_backend}, env://, timeout={init_timeout}s)"
                f" MASTER_ADDR={os.environ.get('MASTER_ADDR')}"
                f":{os.environ.get('MASTER_PORT', '?')}",
                flush=True,
            )
            dist.init_process_group(
                backend=_comm_backend,
                init_method=init_method,
                timeout=timedelta(seconds=init_timeout),
            )
        else:
            print(
                f"[RANK {rank}] init_process_group"
                f"({_comm_backend}, {cfg.distributed.dist_url})...",
                flush=True,
            )
            dist.init_process_group(
                backend=_comm_backend,
                init_method=cfg.distributed.dist_url,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(seconds=120),
            )
    print(
        f"[RANK {rank}] torch.distributed initialized",
        flush=True,
    )

    # ── Step 6: Create ALL process groups (collective) ───
    pp_process_group = None
    pp_ranks_in_group: list[int] = []
    for d in range(dp_size):
        pp_ranks = [
            d * pp_size + s for s in range(pp_size)
        ]
        grp = dist.new_group(ranks=pp_ranks)
        if rank in pp_ranks:
            pp_process_group = grp
            pp_ranks_in_group = pp_ranks

    dp_process_group = None
    dp_ranks_in_group: list[int] = []
    for s in range(pp_size):
        dp_ranks = [
            d * pp_size + s for d in range(dp_size)
        ]
        grp = dist.new_group(ranks=dp_ranks)
        if rank in dp_ranks:
            dp_process_group = grp
            dp_ranks_in_group = dp_ranks

    print(
        f"[RANK {rank}] PP group: {pp_ranks_in_group}"
        f" DP group: {dp_ranks_in_group}",
        flush=True,
    )

    # ── Step 7: CuPy NCCL communicators ──────────────────
    nccl = NCCLBackend(
        rank=rank,
        world_size=world_size,
        cuda_id=cuda_id,
    )
    nccl.setup_communicators(
        pp_rank=pp_rank,
        dp_rank=dp_rank,
        pp_size=pp_size,
        dp_size=dp_size,
        dist_store=None,
    )

    # ── Step 8: Create PipelineStage model ───────────────
    model = PipelineStage(
        model_config=cfg.model,
        start_layer=boundaries[pp_rank],
        end_layer=boundaries[pp_rank + 1],
        is_first=is_first,
        is_last=is_last,
    ).to(device).bfloat16()

    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    print(
        f"[RANK {rank}] layers"
        f" [{boundaries[pp_rank]},{boundaries[pp_rank+1]})"
        f" params={n_params:,}"
        f" weight_mem={param_bytes/1024**3:.2f}GB (bf16)",
        flush=True,
    )

    # ── Step 9: Broadcast weights dp_rank=0 → others ────
    if dp_size > 1 and dp_ranks_in_group:
        for param in model.parameters():
            param_cpu = param.data.cpu()
            dist.broadcast(
                param_cpu,
                src=dp_ranks_in_group[0],
                group=dp_process_group,
            )
            param.data.copy_(param_cpu.to(device))

    # ── Step 10: Re-seed for data diversity ──────────────
    torch.manual_seed(cfg.training.seed * 31 + rank)

    # ── Step 11: Flatten params for DP AllReduce ─────────
    flat_param: Any = None
    if dp_size > 1:
        flat_param = flatten_params(model.parameters())

    # ── Step 12: Optimizer ───────────────────────────────
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [
        p for p in model.parameters() if p.dim() < 2
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": cfg.training.weight_decay,
            },
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
        betas=cfg.training.betas,
    )

    # ── GPU memory report after model + optimizer ────────
    _mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
    _mem_resv = torch.cuda.memory_reserved(device) / 1024**3
    _mem_total = (
        torch.cuda.get_device_properties(device).total_memory
        / 1024**3
    )
    print(
        f"[RANK {rank}] GPU mem after init:"
        f" alloc={_mem_alloc:.2f}GB"
        f" reserved={_mem_resv:.2f}GB"
        f" total={_mem_total:.2f}GB",
        flush=True,
    )

    # ── Step 13: CUDA streams ────────────────────────────
    if gc_controller and gc_controller.is_supported:
        gc_bundle = gc_controller.get_default_bundle()
        comp_stream = gc_bundle.comp
        recv_stream = gc_bundle.recv
        send_stream = gc_bundle.send
        dp_stream = gc_bundle.dp if dp_size > 1 else None
    else:
        comp_stream = torch.cuda.Stream(
            device=device
        )
        recv_stream = torch.cuda.Stream(
            device=device, priority=-1
        )
        send_stream = torch.cuda.Stream(
            device=device, priority=-1
        )
        dp_stream = None
        if dp_size > 1:
            dp_stream = torch.cuda.Stream(
                device=device, priority=-1
            )

    # ── Step 14: Pre-allocate activation buffers ─────────
    num_micro = cfg.training.num_microbatches
    act_shape = (
        cfg.training.micro_batch_size,
        cfg.model.seq_length,
        cfg.model.embedding_dim,
    )
    input_buffers: list[torch.Tensor] | None = None
    if not is_first:
        input_buffers = [
            torch.zeros(
                act_shape,
                dtype=torch.bfloat16,
                requires_grad=True,
                device=device,
            )
            for _ in range(num_micro)
        ]
    grad_buffers: list[torch.Tensor] | None = None
    if not is_last:
        grad_buffers = [
            torch.zeros(
                act_shape,
                dtype=torch.bfloat16,
                requires_grad=False,
                device=device,
            )
            for _ in range(num_micro)
        ]

    # ── Step 15: Pipeline neighbour ranks ────────────────
    # CuPy NCCL communicators are indexed by group-local
    # rank (pp_rank 0..pp_size-1), NOT global rank.
    # Use raw pp_rank offsets directly — the communicator
    # was created with NcclCommunicator(pp_size, uid,
    # pp_rank), so peer pp_rank±1 is the correct target.
    pp_prev = pp_rank - 1 if pp_rank > 0 else None
    pp_next = (
        pp_rank + 1 if pp_rank < pp_size - 1 else None
    )

    # ── Step 16: Data loaders ────────────────────────────
    train_embeds, train_labels = train_data
    val_embeds, val_labels = val_data
    train_loader = SyntheticDataLoader(
        train_embeds,
        train_labels,
        cfg.training.micro_batch_size,
    )

    print(
        f"[RANK {rank}] Ready. micro={num_micro}"
        f" act_shape={act_shape}",
        flush=True,
    )

    # ── Step 17a: Resolve pipeline schedule ──────────────
    use_1f1b = False
    stage_ops: list[tuple[str, int]] | None = None

    sched = plan.schedule_type or "gpipe"
    sched_norm = (
        sched.lower().replace("-", "").replace("_", "")
    )
    use_1f1b = sched_norm == "1f1b"

    if not is_single and pp_size > 1:
        timeline = build_schedule(
            sched, pp_size, num_micro,
        )
        stage_ops = timeline[pp_rank]
        sched_label = "1F1B" if use_1f1b else "GPipe"
        print(
            f"[RANK {rank}] Schedule: {sched_label}"
            f" | {len(stage_ops)} ops for"
            f" stage {pp_rank}",
            flush=True,
        )

    # ── Step 17b: Training loop ──────────────────────────
    _perf_log = os.environ.get("ASTEROID_PERF_LOG", "")
    _perf_log = _perf_log in ("1", "true", "yes")

    # ── Start watchdog daemon ────────────────────────────
    global _wd_ts
    _wd = threading.Thread(
        target=_watchdog_loop, args=(rank,), daemon=True,
    )
    _wd.start()

    dist.barrier()
    model.train()
    t0 = time.time()

    for iter_num in range(cfg.training.max_iters):
        _wd_ts = time.time()           # pet the watchdog
        iter_start = time.time()

        # Green context: update streams for this step
        _gc_prev_sm = None
        if gc_controller and gc_controller.is_supported:
            _gc_sm, _gc_gen = (
                gc_controller.backend.activate_for_step(
                    iter_num
                )
            )
            _gc_prev_sm = _gc_sm
            _gc_b = gc_controller._bundles.get(_gc_sm)
            if _gc_b is not None:
                comp_stream = _gc_b.comp
                recv_stream = _gc_b.recv
                send_stream = _gc_b.send
                if dp_size > 1:
                    dp_stream = _gc_b.dp

        lr = get_lr(iter_num, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        # MUST use set_to_none=False to preserve the
        # flat_param grad storage linkage.  set_to_none=True
        # would detach param.grad from the flat buffer,
        # causing DP all_reduce to operate on stale data.
        optimizer.zero_grad(set_to_none=False)

        # Zero pre-allocated input grads
        if input_buffers is not None:
            for buf in input_buffers:
                if buf.grad is not None:
                    buf.grad.zero_()

        micro_losses: list = []

        # Collect micro-batches
        micro_inputs: list[torch.Tensor] = []
        micro_targets: list[torch.Tensor] = []
        for _ in range(num_micro):
            x, y = train_loader.get_batch()
            micro_inputs.append(x.to(device))
            micro_targets.append(y.to(device))

        # ── Pipeline Execution (schedule-driven) ────────────
        cached: list[torch.Tensor] = [
            torch.tensor(0.0)
        ] * num_micro

        # Whether this stage needs recv for F / B ops.
        _needs_fwd_recv = pp_prev is not None
        _needs_bwd_recv = pp_next is not None

        def _do_forward(
            m: int,
            skip_recv: bool = False,
            start_group_before_send: bool = False,
            end_group_after_recv: bool = False,
        ) -> None:
            if is_single:
                _fwd_single(
                    model,
                    micro_inputs[m],
                    micro_targets[m],
                    cached,
                    micro_losses,
                    num_micro,
                    comp_stream,
                )
                # _fwd_single appends → fix slot
                cached[m] = cached.pop()
            elif is_first:
                out_list: list[torch.Tensor] = []
                _fwd_first(
                    model,
                    micro_inputs[m],
                    out_list,
                    nccl,
                    pp_next,
                    comp_stream,
                    send_stream,
                    start_group_before_send=(
                        start_group_before_send
                    ),
                )
                cached[m] = out_list[0]
            elif is_last:
                assert input_buffers is not None
                out_list2: list[torch.Tensor] = []
                _fwd_last(
                    model,
                    input_buffers[m],
                    micro_targets[m],
                    out_list2,
                    micro_losses,
                    num_micro,
                    nccl,
                    pp_prev,
                    comp_stream,
                    recv_stream,
                    skip_recv=skip_recv,
                    end_group_after_recv=(
                        end_group_after_recv
                    ),
                )
                cached[m] = out_list2[0]
            else:
                assert input_buffers is not None
                out_list3: list[torch.Tensor] = []
                _fwd_middle(
                    model,
                    input_buffers[m],
                    out_list3,
                    nccl,
                    pp_prev,
                    pp_next,
                    comp_stream,
                    recv_stream,
                    send_stream,
                    skip_recv=skip_recv,
                    start_group_before_send=(
                        start_group_before_send
                    ),
                    end_group_after_recv=(
                        end_group_after_recv
                    ),
                )
                cached[m] = out_list3[0]

        def _do_backward(
            m: int,
            skip_recv: bool = False,
            start_group_before_send: bool = False,
            end_group_after_recv: bool = False,
        ) -> None:
            if is_single:
                _bwd_single(cached[m], comp_stream)
            elif is_last:
                assert input_buffers is not None
                _bwd_last(
                    cached[m],
                    input_buffers[m],
                    nccl,
                    pp_prev,
                    comp_stream,
                    send_stream,
                    start_group_before_send=(
                        start_group_before_send
                    ),
                )
            elif is_first:
                assert grad_buffers is not None
                _bwd_first(
                    cached[m],
                    grad_buffers[m],
                    nccl,
                    pp_next,
                    comp_stream,
                    recv_stream,
                    skip_recv=skip_recv,
                    end_group_after_recv=(
                        end_group_after_recv
                    ),
                )
            else:
                assert input_buffers is not None
                assert grad_buffers is not None
                _bwd_middle(
                    cached[m],
                    input_buffers[m],
                    grad_buffers[m],
                    nccl,
                    pp_prev,
                    pp_next,
                    comp_stream,
                    recv_stream,
                    send_stream,
                    skip_recv=skip_recv,
                    start_group_before_send=(
                        start_group_before_send
                    ),
                    end_group_after_recv=(
                        end_group_after_recv
                    ),
                )

        if use_1f1b and stage_ops is not None:
            # ── 1F1B Interleaved Schedule ────────────────
            # NCCL group notes:
            #   Without ncclGroupStart/End, NCCL serialises
            #   ALL P2P ops on the same communicator.
            #   In 1F1B, every F→B transition produces
            #   send(next) then recv(next) — same peer,
            #   opposite direction → deadlock.  Similarly
            #   every B→F has send(prev) then recv(prev).
            #   We wrap these paired ops in a single group
            #   so NCCL resolves them atomically.
            _skip_prepost = (
                has_cupy_nccl()
                and nccl.pp_nccl_comm is not None
            )
            # Whether F/B ops actually have a send or recv
            # to the relevant peer.
            _f_sends_next = (
                pp_next is not None and not is_last
            )
            _b_sends_prev = (
                pp_prev is not None and not is_first
            )

            pending_recv = None  # _PendingRecv | None
            n_ops = len(stage_ops)
            _group_open = False

            for i, (action, m) in enumerate(stage_ops):
                if i == 0 or i == n_ops - 1 or i % 8 == 0:
                    print(
                        f"[R{rank}] {action}{m}"
                        f" ({i+1}/{n_ops})",
                        flush=True,
                    )
                # 1) Complete any pre-posted recv
                if pending_recv is not None:
                    pending_recv.wait()
                    pending_recv = None
                    skip = True
                else:
                    skip = False

                # Determine group flags.
                # start_group: call ncclGroupStart before
                #   this op's SEND (pairs with next op's
                #   recv to the same peer).
                # end_group: call ncclGroupEnd after this
                #   op's RECV (closes a group opened by
                #   the previous op's send).
                next_action = (
                    stage_ops[i + 1][0]
                    if i + 1 < n_ops
                    else None
                )
                sg = False
                if _skip_prepost:
                    if (
                        action == "F"
                        and next_action == "B"
                        and _f_sends_next
                    ):
                        sg = True
                    elif (
                        action == "B"
                        and next_action == "F"
                        and _b_sends_prev
                    ):
                        sg = True
                eg = _group_open

                # 2) Execute current F or B
                if action == "F":
                    _do_forward(
                        m,
                        skip_recv=skip,
                        start_group_before_send=sg,
                        end_group_after_recv=eg,
                    )
                else:
                    _do_backward(
                        m,
                        skip_recv=skip,
                        start_group_before_send=sg,
                        end_group_after_recv=eg,
                    )

                _group_open = sg

                # 3) Pre-post recv for next op (overlap)
                #    Skipped for CuPy NCCL (see note above).
                if (
                    not _skip_prepost
                    and i + 1 < n_ops
                ):
                    na, nm = stage_ops[i + 1]
                    if (
                        na == "F"
                        and _needs_fwd_recv
                        and input_buffers is not None
                    ):
                        pending_recv = (
                            nccl.recv_async(
                                input_buffers[nm],
                                pp_prev,
                            )
                        )
                    elif (
                        na == "B"
                        and _needs_bwd_recv
                        and grad_buffers is not None
                    ):
                        pending_recv = (
                            nccl.recv_async(
                                grad_buffers[nm],
                                pp_next,
                            )
                        )

            # Drain any in-flight async sends before
            # the optimizer step / DP allreduce.
            nccl.flush_sends()
        else:
            # ── GPipe Schedule ───────────────────────────
            for m in range(num_micro):
                _do_forward(m)

            dist.barrier()

            for m in reversed(range(num_micro)):
                _do_backward(m)

        # ── DP AllReduce ─────────────────────────────────
        # Use CuPy NCCL allreduce on the DP communicator.
        # This runs directly on GPU (— no CPU copy) and
        # avoids torch.distributed backend conflicts.
        if (
            dp_size > 1
            and flat_param is not None
            and nccl.dp_nccl_comm is not None
        ):
            nccl.allreduce(flat_param.grad.data)
            flat_param.grad.data.div_(dp_size)

        # ── Grad clip + optimizer step ───────────────────
        if cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.grad_clip,
            )
        optimizer.step()

        # ── GPU mem after optimizer step (first 3 iters) ─
        if iter_num < 3:
            _ma = torch.cuda.memory_allocated(device)
            _mr = torch.cuda.memory_reserved(device)
            _mp = torch.cuda.max_memory_allocated(device)
            print(
                f"[RANK {rank}] iter {iter_num}"
                f" post-step:"
                f" alloc={_ma/1024**3:.2f}GB"
                f" peak={_mp/1024**3:.2f}GB"
                f" reserved={_mr/1024**3:.2f}GB",
                flush=True,
            )

        # Local sync only — pipeline schedule + flush_sends
        # already ensure all P2P is done.  A global barrier
        # here serialises every rank and adds >200 ms/iter.
        torch.cuda.synchronize()

        # Compact timing (every iteration, rank 0 only)
        if rank == 0:
            iter_ms = (time.time() - iter_start) * 1000.0
            print(
                f"[TIMING] iter {iter_num}"
                f" {iter_ms:.0f}ms",
                flush=True,
            )

        # Free cached blocks so the next iteration starts
        # with maximum available memory for activations.
        torch.cuda.empty_cache()

        # ── Per-stage perf diagnostics ───────────────────
        if _perf_log and (
            iter_num % cfg.training.log_interval == 0
        ):
            iter_ms = (
                (time.time() - iter_start) * 1000.0
            )
            print(
                f"[PERF RANK {rank}] iter {iter_num}"
                f" | stage {pp_rank}"
                f" | total={iter_ms:.1f}ms",
                flush=True,
            )

        # Green context: deactivate for this step
        if _gc_prev_sm is not None and gc_controller:
            gc_controller.backend.deactivate(
                _gc_prev_sm
            )
            _gc_prev_sm = None

        # ── Logging ──────────────────────────────────────
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        should_log = (
            (is_last or is_single)
            and dp_rank == 0
        )
        if should_log:
            avg = (
                sum(t.item() if hasattr(t, 'item') else t
                    for t in micro_losses)
                / len(micro_losses)
                if micro_losses
                else 0.0
            )
            print(
                f"  iter {iter_num:>5d} | loss={avg:.4f}"
                f" | lr={lr:.2e} | dt={dt*1000:.1f}ms",
                flush=True,
            )

        # ── Checkpoint ───────────────────────────────────
        should_ckpt = (
            iter_num > 0
            and iter_num % cfg.training.eval_interval == 0
        )
        if should_ckpt:
            _save_checkpoint(
                model, optimizer, iter_num, rank, args
            )

    # ── Flush any in-flight async checkpoint ─────────
    _flush_checkpoint()

    # ── Green context cleanup (before process group) ──
    if gc_controller is not None:
        gc_controller.close()
        gc_controller = None

    # ── Cleanup ──────────────────────────────────────────
    dist.barrier()
    print(
        f"[RANK {rank}] Training complete.",
        flush=True,
    )
    dist.destroy_process_group()


# ── Forward helpers ──────────────────────────────────────


def _fwd_single(
    model: nn.Module,
    micro_input: torch.Tensor,
    micro_target: torch.Tensor,
    cached: list[torch.Tensor],
    micro_losses: list,
    num_micro: int,
    comp_stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(comp_stream):
        loss = model(micro_input, micro_target)
        scaled_loss = loss / num_micro
    micro_losses.append(loss.detach())
    cached.append(scaled_loss)


def _fwd_first(
    model: nn.Module,
    micro_input: torch.Tensor,
    cached: list[torch.Tensor],
    nccl: NCCLBackend,
    pp_next: int | None,
    comp_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
    start_group_before_send: bool = False,
) -> None:
    with torch.cuda.stream(comp_stream):
        out = model(micro_input)
    cached.append(out)
    if pp_next is not None:
        with torch.cuda.stream(send_stream):
            send_stream.wait_stream(comp_stream)
            send_buf = out.data.clone()
            if start_group_before_send:
                nccl_group_start()
            nccl.send(send_buf, pp_next, send_stream)


def _fwd_last(
    model: nn.Module,
    input_buf: torch.Tensor,
    micro_target: torch.Tensor,
    cached: list[torch.Tensor],
    micro_losses: list,
    num_micro: int,
    nccl: NCCLBackend,
    pp_prev: int | None,
    comp_stream: torch.cuda.Stream,
    recv_stream: torch.cuda.Stream,
    skip_recv: bool = False,
    end_group_after_recv: bool = False,
) -> None:
    if pp_prev is not None and not skip_recv:
        with torch.cuda.stream(recv_stream):
            nccl.recv(input_buf, pp_prev, recv_stream)
            if end_group_after_recv:
                nccl_group_end()
    with torch.cuda.stream(comp_stream):
        if pp_prev is not None and not skip_recv:
            comp_stream.wait_stream(recv_stream)
        loss = model(input_buf, micro_target)
        scaled_loss = loss / num_micro
    micro_losses.append(loss.detach())
    cached.append(scaled_loss)


def _fwd_middle(
    model: nn.Module,
    input_buf: torch.Tensor,
    cached: list[torch.Tensor],
    nccl: NCCLBackend,
    pp_prev: int | None,
    pp_next: int | None,
    comp_stream: torch.cuda.Stream,
    recv_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
    skip_recv: bool = False,
    start_group_before_send: bool = False,
    end_group_after_recv: bool = False,
) -> None:
    if pp_prev is not None and not skip_recv:
        with torch.cuda.stream(recv_stream):
            nccl.recv(input_buf, pp_prev, recv_stream)
            if end_group_after_recv:
                nccl_group_end()
    with torch.cuda.stream(comp_stream):
        if pp_prev is not None and not skip_recv:
            comp_stream.wait_stream(recv_stream)
        out = model(input_buf)
    cached.append(out)
    if pp_next is not None:
        with torch.cuda.stream(send_stream):
            send_stream.wait_stream(comp_stream)
            send_buf = out.data.clone()
            if start_group_before_send:
                nccl_group_start()
            nccl.send(send_buf, pp_next, send_stream)


# ── Backward helpers ─────────────────────────────────────


def _bwd_single(
    cached_out: torch.Tensor,
    comp_stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(comp_stream):
        cached_out.backward()


def _bwd_last(
    cached_out: torch.Tensor,
    input_buf: torch.Tensor,
    nccl: NCCLBackend,
    pp_prev: int | None,
    comp_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
    start_group_before_send: bool = False,
) -> None:
    with torch.cuda.stream(comp_stream):
        cached_out.backward()
    if pp_prev is not None and input_buf.grad is not None:
        with torch.cuda.stream(send_stream):
            send_stream.wait_stream(comp_stream)
            grad_clone = input_buf.grad.clone()
            if start_group_before_send:
                nccl_group_start()
            nccl.send(
                grad_clone, pp_prev, send_stream
            )


def _bwd_first(
    cached_out: torch.Tensor,
    grad_buf: torch.Tensor,
    nccl: NCCLBackend,
    pp_next: int | None,
    comp_stream: torch.cuda.Stream,
    recv_stream: torch.cuda.Stream,
    skip_recv: bool = False,
    end_group_after_recv: bool = False,
) -> None:
    if pp_next is not None:
        if not skip_recv:
            with torch.cuda.stream(recv_stream):
                nccl.recv(
                    grad_buf, pp_next, recv_stream
                )
                if end_group_after_recv:
                    nccl_group_end()
        with torch.cuda.stream(comp_stream):
            if not skip_recv:
                comp_stream.wait_stream(recv_stream)
            cached_out.backward(gradient=grad_buf)
    else:
        with torch.cuda.stream(comp_stream):
            cached_out.backward()


def _bwd_middle(
    cached_out: torch.Tensor,
    input_buf: torch.Tensor,
    grad_buf: torch.Tensor,
    nccl: NCCLBackend,
    pp_prev: int | None,
    pp_next: int | None,
    comp_stream: torch.cuda.Stream,
    recv_stream: torch.cuda.Stream,
    send_stream: torch.cuda.Stream,
    skip_recv: bool = False,
    start_group_before_send: bool = False,
    end_group_after_recv: bool = False,
) -> None:
    if pp_next is not None:
        if not skip_recv:
            with torch.cuda.stream(recv_stream):
                nccl.recv(
                    grad_buf, pp_next, recv_stream
                )
                if end_group_after_recv:
                    nccl_group_end()
        with torch.cuda.stream(comp_stream):
            if not skip_recv:
                comp_stream.wait_stream(recv_stream)
            cached_out.backward(gradient=grad_buf)
    else:
        with torch.cuda.stream(comp_stream):
            cached_out.backward()
    if pp_prev is not None and input_buf.grad is not None:
        with torch.cuda.stream(send_stream):
            send_stream.wait_stream(comp_stream)
            grad_clone = input_buf.grad.clone()
            if start_group_before_send:
                nccl_group_start()
            nccl.send(
                grad_clone, pp_prev, send_stream
            )


# ── Utilities ────────────────────────────────────────────


def _resolve_cuda_id(
    rank: int,
    plan: ParallelismPlan,
    pp_size: int,
) -> int:
    """Map global rank → CUDA device id via plan."""
    if not plan.device_groups:
        return rank
    gpu_map: dict[int, int] = {}
    for stage_idx, devices in plan.device_groups.items():
        for dp_idx, dev_id in enumerate(devices):
            global_rank = dp_idx * pp_size + stage_idx
            gpu_map[global_rank] = dev_id
    return gpu_map.get(rank, rank)

def _mps_refcount_acquire(
    rank: int,
    cfg: BaseConfig,
    plan: ParallelismPlan,
) -> bool:
    """Atomically start MPS if first worker on this GPU.

    Returns True if this call started the daemon.
    """
    from baselines.utils.mps import (
        _resolve_pipe_dir,
        start_mps,
    )

    pp_size = cfg.parallel.pp_size
    cuda_id = _resolve_cuda_id(rank, plan, pp_size)
    pipe_dir = _resolve_pipe_dir(cuda_id, cfg.device)
    os.makedirs(pipe_dir, exist_ok=True)
    lock_path = os.path.join(pipe_dir, ".mps.lock")
    ref_path = os.path.join(pipe_dir, ".mps.refcount")

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            n = 0
            if os.path.exists(ref_path):
                with open(ref_path) as rf:
                    n = int(rf.read().strip() or "0")
            started = False
            if n == 0:
                start_mps(cuda_id, cfg.device)
                started = True
            with open(ref_path, "w") as wf:
                wf.write(str(n + 1))
            return started
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _mps_refcount_release(
    rank: int,
    cfg: BaseConfig,
    plan: ParallelismPlan,
) -> None:
    """Atomically stop MPS if last worker on this GPU."""
    from baselines.utils.mps import (
        _resolve_pipe_dir,
        stop_mps,
    )

    pp_size = cfg.parallel.pp_size
    cuda_id = _resolve_cuda_id(rank, plan, pp_size)
    pipe_dir = _resolve_pipe_dir(cuda_id, cfg.device)
    lock_path = os.path.join(pipe_dir, ".mps.lock")
    ref_path = os.path.join(pipe_dir, ".mps.refcount")

    if not os.path.exists(lock_path):
        return

    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            n = 1
            if os.path.exists(ref_path):
                with open(ref_path) as rf:
                    n = int(rf.read().strip() or "1")
            n = max(0, n - 1)
            if n == 0:
                stop_mps(cuda_id)
                try:
                    os.remove(ref_path)
                except OSError:
                    pass
            else:
                with open(ref_path, "w") as wf:
                    wf.write(str(n))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

# Global handle for the in-flight async checkpoint thread.
_ckpt_thread: threading.Thread | None = None


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    rank: int,
    args: argparse.Namespace,
) -> None:
    """Async per-rank checkpoint save.

    Snapshots state_dicts to CPU (fast) on the calling
    thread, then writes to disk in a background thread
    so training can continue immediately.
    """
    global _ckpt_thread

    # Wait for any previous async save to finish before
    # snapshotting new state (avoids concurrent writes).
    if _ckpt_thread is not None:
        _ckpt_thread.join()
        _ckpt_thread = None

    # Snapshot to CPU — this is the fast part.
    model_sd = {
        k: v.cpu() for k, v in model.state_dict().items()
    }
    optim_sd = optimizer.state_dict()
    # Deep-copy optimizer state tensors to CPU so the
    # background thread owns all data.
    cpu_optim_sd = _cpu_optim_state(optim_sd)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(
        ckpt_dir, f"rank{rank}_iter{iter_num}.pt"
    )

    def _write() -> None:
        torch.save(
            {
                "model": model_sd,
                "optimizer": cpu_optim_sd,
                "iter": iter_num,
            },
            path,
        )
        print(
            f"  [R{rank}] Checkpoint written: {path}",
            flush=True,
        )

    _ckpt_thread = threading.Thread(
        target=_write, daemon=True,
    )
    _ckpt_thread.start()


def _cpu_optim_state(
    sd: dict,
) -> dict:
    """Deep-copy optimizer state dict with all tensors on CPU."""
    import copy
    out = copy.copy(sd)
    if "state" in out:
        new_state = {}
        for k, v in out["state"].items():
            new_v = {}
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    new_v[sk] = sv.cpu()
                else:
                    new_v[sk] = sv
            new_state[k] = new_v
        out["state"] = new_state
    return out


def _flush_checkpoint() -> None:
    """Wait for any in-flight async checkpoint to finish."""
    global _ckpt_thread
    if _ckpt_thread is not None:
        _ckpt_thread.join()
        _ckpt_thread = None


# ── Strategy factory ─────────────────────────────────────


_STRATEGY_MAP = {
    "dtfm": DTFMStrategy,
    "asteroid": AsteroidStrategy,
    "confident": ConfidentStrategy,
}


def _build_strategy(
    name: str,
    cfg: BaseConfig,
) -> DTFMStrategy | AsteroidStrategy | ConfidentStrategy:
    """Build strategy with constructor matching its signature."""
    pp = cfg.parallel.pp_size
    dp = cfg.parallel.dp_size
    if name == "dtfm":
        return DTFMStrategy(
            pp_size=pp,
            dp_size=dp,
            global_batch_size=(
                cfg.training.global_batch_size
            ),
            micro_batch_size=(
                cfg.training.micro_batch_size
            ),
        )
    if name == "asteroid":
        return AsteroidStrategy(
            num_stages=pp,
            micro_batch_size=cfg.training.micro_batch_size,
            num_microbatches=cfg.training.num_microbatches,
        )
    if name == "confident":
        return ConfidentStrategy(pp_size=pp, dp_size=dp)
    raise ValueError(
        f"Unknown strategy: {name}."
        f" Choose from {list(_STRATEGY_MAP)}"
    )

def _build_topology(
    world_size: int,
) -> DeviceTopology:
    """Create synthetic device topology."""
    specs = [
        DeviceConfig(
            device_id=i,
            compute_capacity=1.0,
            memory_budget_mb=8192.0,
        )
        for i in range(world_size)
    ]
    bw = {
        (i, j): 100.0
        for i in range(world_size)
        for j in range(world_size)
        if i != j
    }
    lat = {
        (i, j): 0.1
        for i in range(world_size)
        for j in range(world_size)
        if i != j
    }
    return DeviceTopology(
        device_specs=specs, bandwidths=bw, latencies=lat
    )


# ── CLI + main ───────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baselines distributed training",
    )
    p.add_argument(
        "--config",
        type=str,
        default="baselines/configs/dtfm_default.yaml",
        help="YAML config path",
    )
    p.add_argument(
        "--strategy",
        type=str,
        choices=list(_STRATEGY_MAP),
        default="dtfm",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, no training",
    )
    p.add_argument(
        "--spawn",
        action="store_true",
        help="Use mp.spawn (single-node convenience)",
    )
    p.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="GPU count (--spawn mode only)",
    )
    p.add_argument(
        "--dist-url",
        type=str,
        default="tcp://127.0.0.1:29500",
    )
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Global rank (per-process mode)",
    )
    p.add_argument(
        "--world-size",
        type=int,
        default=-1,
        help="Total ranks (per-process mode)",
    )
    p.add_argument(
        "--cuda-id",
        type=int,
        default=-1,
        help="GPU id (per-process mode)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./output",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
    )
    p.add_argument(
        "--enable-mps",
        action="store_true",
        help="Enable NVIDIA MPS for GPU workers",
    )
    p.add_argument(
        "--mps-thread-pct",
        type=int,
        default=None,
        help="MPS active thread percentage (1-100)",
    )
    # ── Green context args ─────────────────────────────
    p.add_argument(
        "--greenctx-enabled",
        action="store_true",
        help="Enable green context SM partitioning",
    )
    p.add_argument(
        "--greenctx-backend",
        type=str,
        choices=["auto", "cpp", "torch_native", "off"],
        default="auto",
    )
    p.add_argument(
        "--greenctx-trace",
        type=str,
        default=None,
        help="Path to green context trace file",
    )
    p.add_argument(
        "--greenctx-clock-mode",
        type=str,
        choices=["wall", "step"],
        default="step",
    )
    p.add_argument(
        "--greenctx-strict",
        action="store_true",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for distributed training."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(name)s"
            " %(levelname)s %(message)s"
        ),
    )

    # ── Load config ──────────────────────────────────────
    cfg = load_config(args.config)
    seed_everything(cfg.training.seed)

    # ── MPS CLI overrides ──────────────────────────────
    if args.enable_mps:
        cfg.device.mps_enabled = True
    if args.mps_thread_pct is not None:
        cfg.device.mps_active_thread_percentage = (
            args.mps_thread_pct
        )
    if cfg.device.mps_enabled:
        cfg.device.validate()

    # ── Green context CLI overrides ────────────────────
    if args.greenctx_enabled:
        cfg.greenctx.enabled = True
    if args.greenctx_backend:
        cfg.greenctx.backend = args.greenctx_backend
    if args.greenctx_trace:
        cfg.greenctx.trace_path = args.greenctx_trace
    if args.greenctx_clock_mode:
        cfg.greenctx.clock_mode = args.greenctx_clock_mode
    if args.greenctx_strict:
        cfg.greenctx.strict = True
    # ── Resolve world_size ───────────────────────────────
    if args.spawn:
        world_size = args.num_gpus
    elif args.world_size > 0:
        world_size = args.world_size
    else:
        # torchrun sets env vars
        world_size = int(
            os.environ.get("WORLD_SIZE", "1")
        )

    # Override config from CLI
    cfg.distributed.world_size = world_size
    cfg.distributed.dist_url = args.dist_url
    # Ensure pp_size * dp_size == world_size
    pp = cfg.parallel.pp_size
    dp = cfg.parallel.dp_size
    if pp * dp != world_size:
        dp = max(1, world_size // pp)
        cfg.parallel.dp_size = dp
        cfg.parallel.world_size = world_size
        logger.info(
            "Adjusted dp_size=%d for world_size=%d",
            dp,
            world_size,
        )

    # ── Create strategy + plan ───────────────────────────
    # Prefer offline HPP plan (from deploy pipeline)
    # over inline strategy auto-search.
    hpp_plan_path = os.environ.get(
        "HPP_PLAN_PATH", ""
    )
    plan: ParallelismPlan | None = None
    if hpp_plan_path and os.path.isfile(hpp_plan_path):
        try:
            with open(hpp_plan_path) as f:
                plan_data = json.load(f)
            plan = ParallelismPlan.from_json(plan_data)
            logger.info(
                "Loaded offline plan from %s: "
                "stages=%d partition=%s",
                hpp_plan_path,
                len(plan.partition_points) + 1,
                plan.partition_points,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s), "
                "falling back to inline plan",
                hpp_plan_path,
                exc,
            )
            plan = None

    if plan is None:
        strategy = _build_strategy(
            args.strategy, cfg
        )
        topology = _build_topology(world_size)
        plan = strategy.create_plan(
            cfg.model, topology
        )

    # The auto-search may have chosen a different
    # (pp, dp) factorisation.  Update config to match.
    plan_pp = max(1, len(plan.device_groups))
    plan_dp = max(1, world_size // plan_pp)
    if (
        plan_pp != cfg.parallel.pp_size
        or plan_dp != cfg.parallel.dp_size
    ):
        cfg.parallel.pp_size = plan_pp
        cfg.parallel.dp_size = plan_dp
        cfg.parallel.num_stages = plan_pp
        logger.info(
            "Plan selected PP=%d DP=%d",
            plan_pp,
            plan_dp,
        )

    logger.info(
        "Plan: partition=%s schedule=%s latency=%.2fms",
        plan.partition_points,
        plan.schedule_type,
        plan.estimated_latency_ms,
    )

    # ── Print banner (after plan so PP/DP reflect
    #    auto-search result) ──────────────────────────────
    print(f"\n{'=' * 56}")
    print("  Baselines Distributed Training")
    print(
        f"  strategy={args.strategy}"
        f"  world_size={world_size}"
        f"  PP={cfg.parallel.pp_size}"
        f"  DP={cfg.parallel.dp_size}"
    )
    print(
        f"  model={cfg.model.model_type}"
        f"  layers={cfg.model.num_layers}"
        f"  dim={cfg.model.embedding_dim}"
    )
    print(
        f"  micro_batch={cfg.training.micro_batch_size}"
        f"  num_micro={cfg.training.num_microbatches}"
    )
    print(
        f"  partition={plan.partition_points}"
    )
    print(f"{'=' * 56}\n")

    if args.dry_run:
        print("Dry-run complete. Plan:")
        print(
            f"  partition_points: {plan.partition_points}"
        )
        print(f"  device_groups: {plan.device_groups}")
        print(f"  schedule: {plan.schedule_type}")
        print(
            f"  latency: {plan.estimated_latency_ms:.2f}ms"
        )
        return

    # ── Generate synthetic data ──────────────────────────
    seq_len = cfg.model.seq_length
    vocab = cfg.model.vocab_size
    n_cls = cfg.model.num_classes
    task = cfg.model.task_type
    print("Generating synthetic dataset...", flush=True)
    if task == "lm":
        # LM: token IDs in, token IDs as target
        train_x = torch.randint(0, vocab, (2000, seq_len))
        train_y = torch.randint(0, vocab, (2000, seq_len))
        val_x = torch.randint(0, vocab, (200, seq_len))
        val_y = torch.randint(0, vocab, (200, seq_len))
    else:
        # Classification: token IDs in, class labels
        train_x = torch.randint(0, vocab, (2000, seq_len))
        train_y = torch.randint(0, n_cls, (2000,))
        val_x = torch.randint(0, vocab, (200, seq_len))
        val_y = torch.randint(0, n_cls, (200,))
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)

    # ── Launch ───────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    if args.spawn:
        import torch.multiprocessing as mp

        mp.set_start_method("fork", force=True)

        # Start MPS daemons before spawning workers.
        if cfg.device.mps_enabled:
            gpu_ids = list(range(world_size))
            for gid in gpu_ids:
                start_mps(gid, cfg.device)

        print(
            f"Spawning {world_size} workers"
            " (mp.spawn)...",
            flush=True,
        )
        try:
            mp.spawn(
                worker,
                args=(
                    cfg,
                    plan,
                    train_data,
                    val_data,
                    args,
                ),
                nprocs=world_size,
                join=True,
            )
        finally:
            if cfg.device.mps_enabled:
                stop_all_mps()
    else:
        # Per-process mode (torchrun / SSH)
        rank = args.rank
        if rank == 0:
            # torchrun sets LOCAL_RANK / RANK
            env_rank = os.environ.get("RANK")
            if env_rank is not None:
                rank = int(env_rank)
        if args.cuda_id >= 0:
            pass  # user-specified, handled in worker

        # Per-process MPS: use filelock refcount so the
        # first worker on a GPU starts the daemon and
        # the last worker stops it.
        _mps_started = False
        if cfg.device.mps_enabled:
            _mps_started = _mps_refcount_acquire(
                rank, cfg, plan
            )

        try:
            worker(
                rank,
                cfg,
                plan,
                train_data,
                val_data,
                args,
            )
        finally:
            if cfg.device.mps_enabled:
                _mps_refcount_release(
                    rank, cfg, plan
                )

    print("\nAll workers finished.", flush=True)


if __name__ == "__main__":
    main()
