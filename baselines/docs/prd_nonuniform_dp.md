# PRD: Non-Uniform Data Parallelism Runtime Support

## Problem
The Asteroid scheduler produces plans with **non-uniform DP** — different
pipeline stages can have different numbers of device replicas.  Example from
current `hpp_plan.json`:

```
device_groups: {0:[0], 1:[1], 2:[2], 3:[3,4], 4:[5]}
```

Stage 3 has 2 devices (DP=2), all others have 1 device (DP=1).

The `train.py` runtime assumes a **uniform PP×DP grid**:
- `pp_rank = rank % pp_size` → wrong mapping
- `dp_rank = rank // pp_size` → wrong mapping
- PP groups: uniform column formula → orphans DP replicas
- Fixed `pp_prev`/`pp_next` → can't route micro-batches to varying peers

Result: rank 5 gets `PP group: [] DP group: []` — isolated.  All other
ranks timeout on `init_process_group`.

## Solution: Per-Micro-Batch Routing + World NCCL Communicator

### 1. Rank Mapping from device_groups
- `rank_to_stage[rank]` = stage index (replaces `rank % pp_size`)
- `rank_to_dp_idx[rank]` = position within stage (replaces `rank // pp_size`)
- `stage_dp_size[stage]` = len(device_group) per stage

### 2. World-level NCCL Communicator for PP P2P
- Create a single NCCL communicator with ALL ranks (comm_rank = global rank)
- P2P send/recv uses global ranks as targets
- Keep per-stage DP communicator for allreduce

### 3. Per-Micro-Batch Routing Tables
- `fwd_recv_source[micro_id]` → global rank to receive activation from
- `fwd_send_target[micro_id]` → global rank to send activation to
- Backward uses reversed routing (recv from send target, send to recv source)
- Built from `device_groups` + `micro_batch_alloc`

### 4. Schedule Filtering
- DP replicas at stages with dp_size > 1 only process assigned micro-batches
- Filter schedule ops to `my_micro_ids` subset

### 5. Process Group Creation
- PP groups: built from device_groups columns (for barriers/bookkeeping)
- DP groups: built from device_groups stages (for weight broadcast + allreduce)

## Files Changed
1. `baselines/communication/nccl.py` — add `setup_nonuniform_communicators()`
2. `baselines/train.py` — add routing helpers, modify `_worker_impl()`
3. `baselines/core/config.py` — no changes (ParallelismPlan already has device_groups)

## Backward Compatibility
- Uniform DP plans (same device count per stage) use existing code path unchanged
- `_is_nonuniform_dp(plan)` flag gates the new logic
