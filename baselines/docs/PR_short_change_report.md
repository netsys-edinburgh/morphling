# PR Short Change Report

**Range:** `5c5c000b1da6b56356fa9ef6defdf59e219e4903..a28e6f6`  
**Commits:** 49

## What changed (short)

- **Deployment automation hardened**
  - Auto-generates and consistently refreshes inventory.
  - Integrates end-to-end deploy flow (`k3s`/registry/GPU/MPS/profile/build).
  - Improves worker join/setup reliability and node naming.

- **Scheduler and planner improvements**
  - Adds memory-aware Confident scheduling support (including GQA updates).
  - Refactors planner strategy dispatch/profile adapter paths.
  - Uses pairwise bandwidth for better inter-stage comm cost modeling.

- **Training/runtime stability upgrades**
  - Improves 1F1B + NCCL behavior for multi-node PP training.
  - Adds CuPy NCCL allreduce path and related runtime fixes.

- **Profiling robustness**
  - Multiple iperf3 reliability fixes (server-busy/crash handling, backoff, strict iperf3 mode, shell compatibility).

- **Experiment pipeline fixes**
  - Adds per-rank JSONL metrics collector and richer PP/DP timing capture.
  - Fixes stale metrics accumulation (`stop` before clear, overwrite JSONL each run).
  - Updates experiment runner to ensure build/push behavior is handled in-flow.
  - Strengthens stitcher with schema bootstrap + safe metrics-table migration.

- **Reporting and docs**
  - Adds/updates schema and deployment docs.
  - Refreshes scheduler comparison report with comm breakdown columns.

## Key outcomes

- More reliable cluster deployment and worker bring-up.
- Better scheduler/partitioning behavior under heterogeneous resource limits.
- More trustworthy communication timing and stitched experiment analytics.
- Cleaner run-to-run experiment data handling and reproducibility.

## Notes for reviewers

- This range includes generated artifacts and experiment outputs in addition to code changes.
- Highest-impact areas to review first: deploy scripts/playbooks, scheduler logic, training communication path, stitch/schema handling.
