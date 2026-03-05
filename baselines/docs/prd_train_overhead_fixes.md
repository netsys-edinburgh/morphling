# PRD: Training Engineering Overhead Fixes

**Branch:** `fix/train-engineering-overhead`
**Date:** 2025-03-05
**Context:** LLaMA-7B 6-node PP training: 31s/iter actual vs ~8s estimated (4× gap)

## Problem

73% of per-iteration time is engineering overhead, not hardware-bound:

1. **Synchronous barriers** — `dist.barrier()` + `torch.cuda.synchronize()` every iteration (~200ms+ per barrier × 2 = 400ms+)
2. **Blocking recv** — `_dist_recv()` uses synchronous `dist.recv()`, preventing compute/comm overlap
3. **No pre-posted receives** — recv is issued at consumption time, not ahead-of-time
4. **Rank 2 anomaly** — 1.5× slower than peers (42ms vs 27ms/layer); no diagnostic tooling
5. **iperf rank 1→2 failure** — profile shows N/A bandwidth; iperf server may not have been running on rank 2's node

## Changes

### 1. Remove unnecessary per-iteration barriers (`train.py`)
- Remove `dist.barrier()` at top of iteration loop (line 449)
- Remove `torch.cuda.synchronize()` + `dist.barrier()` after optimizer step (lines 634-635)
- Keep sync only for logging accuracy (local-only, no global barrier)
- Keep final cleanup barrier (line 681) and GPipe mid-loop barrier (line 605)

### 2. Async recv with deferred wait (`nccl_functional.py`)
- Add `_dist_recv_async()` returning `(Work, cpu_buf)` handle
- Add `complete_recv()` to wait + copy back to GPU
- Keeps original `_dist_recv()` as fallback for callers that don't need overlap

### 3. Overlap recv with prior compute in 1F1B (`train.py`)
- In the 1F1B loop, pre-post the recv for the NEXT operation while executing the current
- For middle/last stages: post irecv for micro-batch M+1 during compute of micro-batch M
- New `_pre_post_recv()` / `_complete_recv()` helper pair
- Send remains async (already uses `isend` in fallback)

### 4. Fix iperf profiling for rank 1→2 (`profile_and_gather.yaml`)
- Add retry logic for iperf3 network tests
- Add explicit iperf3 server readiness check on all nodes before profiling starts

### 5. Rank 2 diagnostic logging (`train.py`)
- Add per-iteration per-stage compute timing (guarded by `ASTEROID_PERF_LOG=1`)
- Log recv wait time vs compute time to identify bottleneck

## Non-goals
- Switching from gloo to NCCL backend (requires CuPy install; separate effort)
- Changing the partition or planner
- Fixing ProfilerBlock architecture mismatch (separate PR)
