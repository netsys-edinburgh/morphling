# PRD: Experiment Data Schema for Scheduler Evaluation

## 1. Context

Evaluate three scheduling algorithms (DT-FM, Confident, Asteroid) under varying per-node MPS constraints ([20%, 40%, 60%, 80%]). The schema must associate scheduler identity and per-node MPS config with load, execution, and communication metrics at both global and per-rank granularity.

**Sweep space:** 3 schedulers × unique random MPS assignments per node from [20, 40, 60, 80]%.
Each experiment is identified by a unique `(scheduler, R0%, R1%, R2%, R3%, R4%, R5%)` tuple.

## 2. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Normalized relational tables** | Avoid data duplication; enable SQL joins for cross-experiment comparison. |
| **Raw timestamps only during training** | Only capture `start_ts` and `end_ts`; derived fields (`duration_ms`, `straggler_ratio`, etc.) are computed post-training to avoid affecting training behaviour. |
| **Local-only logging** | Each node writes its own JSONL file during training — zero cross-node transmission. Data is stitched post-training. |
| **Reproducible from DB** | The `experiments` table snapshots exact settings so any result row is traceable to its run conditions. |
| **Communication split by type** | PP send/recv and DP allreduce are separate columns. Directly answers "is the bottleneck in pipeline comms or gradient sync?" |

## 3. Data Collection Architecture

### During Training (per node)
- Each rank writes a local JSONL file: `/tmp/asteroid_metrics/rank_{rank}.jsonl`
- Records are appended one-per-line — minimal overhead (one `json.dumps` + file write)
- **Raw values only**: timestamps, byte counts, memory readings — NO derived metrics
- Rank 0 additionally logs `global_step_raw` records (loss, lr, iter wall time from its own perspective)

### Post-Training
1. Experiment script collects JSONL files from all nodes via `scp`
2. Post-processor reads all JSONL files, computes derived metrics (`duration_ms`, `straggler_ratio`, aggregations), and bulk-inserts into SQLite
3. `global_step_metrics` cross-rank columns are populated from the stitched per-rank data

### Record Types (JSONL)
```json
{"type": "load", "rank": 0, "model_load_start_ts": "...", "model_load_end_ts": "...", "memory_after_load_mb": 1850.5}
{"type": "step", "rank": 0, "iter": 5, "start_ts": "...", "end_ts": "...", "forward_start_ts": "...", "forward_end_ts": "...", ...}
{"type": "global_step_raw", "iter": 5, "loss": 0.89, "lr": 6e-06, "iter_start_ts": "...", "iter_end_ts": "..."}
```

## 4. Schema (SQL — SQLite compatible)

```sql
-- ═══════════════════════════════════════════════════════════════════
-- TABLE 1: experiments
-- One row per training run. Top-level identity + config snapshot.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE experiments (
    experiment_id       TEXT PRIMARY KEY,              -- UUID v4
    scheduler           TEXT NOT NULL                  -- 'dtfm' | 'confident' | 'asteroid'
                        CHECK (scheduler IN ('dtfm', 'confident', 'asteroid')),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    failure_reason      TEXT,                          -- NULL unless status='failed'
    git_commit          TEXT,
    docker_image        TEXT,

    -- Model config
    model_name          TEXT NOT NULL,                 -- e.g. 'opt-1.3b'
    model_type          TEXT NOT NULL,                 -- 'gpt2' | 'llama' | 'opt'
    num_layers          INTEGER NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    num_heads           INTEGER NOT NULL,
    d_ff                INTEGER NOT NULL,
    vocab_size          INTEGER NOT NULL,
    max_seq_len         INTEGER NOT NULL,

    -- Training config
    global_batch_size   INTEGER NOT NULL,
    micro_batch_size    INTEGER NOT NULL,
    num_microbatches    INTEGER NOT NULL,
    lr                  REAL NOT NULL,
    min_lr              REAL NOT NULL,
    weight_decay        REAL NOT NULL,
    max_iters           INTEGER NOT NULL,
    warmup_iters        INTEGER NOT NULL,
    grad_clip           REAL NOT NULL,
    dataset             TEXT NOT NULL,
    synthetic_samples   INTEGER NOT NULL DEFAULT 2000,

    -- Parallelism config
    schedule_type       TEXT NOT NULL,                 -- '1f1b' | 'gpipe'
    pp_size             INTEGER NOT NULL,
    dp_size             INTEGER NOT NULL,
    world_size          INTEGER NOT NULL,
    comm_backend        TEXT NOT NULL                  -- 'torch_dist' | 'nccl' | 'gloo'
);

-- ═══════════════════════════════════════════════════════════════════
-- TABLE 2: rank_config
-- Per-rank MPS assignment and node identity for each experiment.
-- This is the independent variable being swept.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE rank_config (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    rank                INTEGER NOT NULL,
    hostname            TEXT NOT NULL,
    ip                  TEXT NOT NULL,
    gpu_id              INTEGER NOT NULL DEFAULT 0,
    pp_stage            INTEGER NOT NULL,
    dp_group            INTEGER NOT NULL,
    mps_pct             INTEGER NOT NULL              -- 20 | 40 | 60 | 80
                        CHECK (mps_pct IN (20, 40, 60, 80)),
    memory_limit_mb     INTEGER NOT NULL,             -- floor(gpu_mem * mps_pct / 100)
    PRIMARY KEY (experiment_id, rank)
);

-- ═══════════════════════════════════════════════════════════════════
-- TABLE 3: load_metrics
-- One row per rank per experiment. Model load + init timing.
-- Raw timestamps logged during training; duration_ms derived post-training.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE load_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    model_load_start_ts     TIMESTAMP,                -- raw (logged during training)
    model_load_end_ts       TIMESTAMP,                -- raw (logged during training)
    model_load_duration_ms  REAL,                     -- DERIVED post-training: end - start
    nccl_init_start_ts      TIMESTAMP,                -- raw
    nccl_init_end_ts        TIMESTAMP,                -- raw
    nccl_init_duration_ms   REAL,                     -- DERIVED post-training
    memory_after_load_mb    REAL,                     -- raw
    PRIMARY KEY (experiment_id, rank),
    FOREIGN KEY (experiment_id, rank) REFERENCES rank_config(experiment_id, rank)
);

-- ═══════════════════════════════════════════════════════════════════
-- TABLE 4: step_metrics
-- One row per (experiment, rank, iteration). The bulk data table.
-- Raw timestamps logged during training. Most legacy `*_ms` values are
-- still derived post-training; selected cumulative `*_ms` values are
-- now logged directly from CUDA events.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE step_metrics (
    experiment_id       TEXT NOT NULL,
    rank                INTEGER NOT NULL,
    iter                INTEGER NOT NULL,

    -- Timing (raw timestamps logged during training)
    start_ts            TIMESTAMP,
    end_ts              TIMESTAMP,
    duration_ms         REAL,                         -- DERIVED post-training

    -- Compute breakdown (raw timestamps logged; durations derived)
    forward_start_ts    TIMESTAMP,
    forward_end_ts      TIMESTAMP,
    forward_ms          REAL,                         -- DERIVED
    backward_start_ts   TIMESTAMP,
    backward_end_ts     TIMESTAMP,
    backward_ms         REAL,                         -- DERIVED
    optimizer_start_ts  TIMESTAMP,
    optimizer_end_ts    TIMESTAMP,
    optimizer_step_ms   REAL,                         -- DERIVED

    -- Pipeline-parallel communication (raw timestamps + byte counts)
    pp_send_start_ts    TIMESTAMP,
    pp_send_end_ts      TIMESTAMP,
    pp_send_ms          REAL,                         -- DERIVED
    pp_recv_start_ts    TIMESTAMP,
    pp_recv_end_ts      TIMESTAMP,
    pp_recv_ms          REAL,                         -- DERIVED
    pp_send_cumulative_ms REAL,                       -- RAW (CUDA events sum)
    pp_recv_cumulative_ms REAL,                       -- RAW (CUDA events sum)
    forward_compute_ms  REAL,                         -- RAW (CUDA events sum)
    backward_compute_ms REAL,                         -- RAW (CUDA events sum)
    pp_send_count       INTEGER,                      -- raw
    pp_recv_count       INTEGER,                      -- raw
    pp_send_bytes       BIGINT,                       -- raw
    pp_recv_bytes       BIGINT,                       -- raw

    -- Data-parallel communication (raw)
    dp_allreduce_start_ts   TIMESTAMP,
    dp_allreduce_end_ts     TIMESTAMP,
    dp_allreduce_ms     REAL,                         -- DERIVED
    dp_allreduce_bytes  BIGINT,                       -- raw

    -- Observables (raw)
    loss                REAL,                         -- NULL for non-last-stage ranks
    lr                  REAL,
    memory_allocated_mb REAL,
    memory_reserved_mb  REAL,
    memory_peak_mb      REAL,

    PRIMARY KEY (experiment_id, rank, iter),
    FOREIGN KEY (experiment_id, rank) REFERENCES rank_config(experiment_id, rank)
);

-- ═══════════════════════════════════════════════════════════════════
-- TABLE 5: global_load_metrics
-- One row per experiment. DERIVED post-training from load_metrics.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE global_load_metrics (
    experiment_id               TEXT PRIMARY KEY REFERENCES experiments(experiment_id),

    overall_load_start_ts       TIMESTAMP,            -- DERIVED: MIN(model_load_start_ts)
    overall_load_end_ts         TIMESTAMP,            -- DERIVED: MAX(model_load_end_ts)
    overall_load_duration_ms    REAL,                  -- DERIVED: end - start

    slowest_rank                INTEGER,               -- DERIVED
    slowest_rank_load_ms        REAL,                  -- DERIVED
    fastest_rank                INTEGER,               -- DERIVED
    fastest_rank_load_ms        REAL,                  -- DERIVED

    overall_nccl_init_ms        REAL,                  -- DERIVED: MAX(nccl_init_duration_ms)
    total_memory_after_load_mb  REAL                   -- DERIVED: SUM
);

-- ═══════════════════════════════════════════════════════════════════
-- TABLE 6: global_step_metrics
-- One row per iteration.
-- During training: rank 0 logs raw values (loss, lr, iter timestamps).
-- Post-training: cross-rank columns populated from stitched data.
-- ═══════════════════════════════════════════════════════════════════
CREATE TABLE global_step_metrics (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    iter                INTEGER NOT NULL,

    -- Raw (logged by rank 0 during training)
    iter_start_ts       TIMESTAMP,                    -- rank 0's wall clock at iter start
    iter_end_ts         TIMESTAMP,                    -- rank 0's wall clock at iter end
    loss                REAL,                         -- from last-stage rank
    lr                  REAL,

    -- DERIVED post-training (from stitched per-rank data)
    iter_duration_ms    REAL,                         -- DERIVED: MAX(end_ts) - MIN(start_ts) across ranks

    total_forward_ms    REAL,                         -- DERIVED: SUM(forward_ms)
    total_backward_ms   REAL,                         -- DERIVED: SUM(backward_ms)
    total_optimizer_ms  REAL,                         -- DERIVED: SUM(optimizer_step_ms)

    total_pp_send_ms    REAL,                         -- DERIVED: SUM(pp_send_ms)
    total_pp_recv_ms    REAL,                         -- DERIVED: SUM(pp_recv_ms)
    total_pp_send_cumulative_ms REAL,                 -- DERIVED: SUM(pp_send_cumulative_ms)
    total_pp_recv_cumulative_ms REAL,                 -- DERIVED: SUM(pp_recv_cumulative_ms)
    total_pp_bytes      BIGINT,                       -- DERIVED: SUM(pp_send_bytes)
    max_pp_send_ms      REAL,                         -- DERIVED: MAX(pp_send_ms)
    max_pp_recv_ms      REAL,                         -- DERIVED: MAX(pp_recv_ms)
    max_pp_send_cumulative_ms REAL,                   -- DERIVED: MAX(pp_send_cumulative_ms)
    max_pp_recv_cumulative_ms REAL,                   -- DERIVED: MAX(pp_recv_cumulative_ms)
    total_forward_compute_ms REAL,                    -- DERIVED: SUM(forward_compute_ms)
    total_backward_compute_ms REAL,                   -- DERIVED: SUM(backward_compute_ms)

    total_dp_allreduce_ms   REAL,                     -- DERIVED: SUM
    max_dp_allreduce_ms     REAL,                     -- DERIVED: MAX
    dp_allreduce_bytes      BIGINT,                   -- DERIVED

    slowest_rank        INTEGER,                      -- DERIVED: rank with MAX(duration_ms)
    fastest_rank        INTEGER,                      -- DERIVED: rank with MIN(duration_ms)
    straggler_ratio     REAL,                         -- DERIVED: MAX/MIN

    PRIMARY KEY (experiment_id, iter)
);

-- ═══════════════════════════════════════════════════════════════════
-- INDEXES for common query patterns
-- ═══════════════════════════════════════════════════════════════════

CREATE INDEX idx_experiments_scheduler ON experiments(scheduler);
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_step_metrics_exp_rank ON step_metrics(experiment_id, rank);
CREATE INDEX idx_step_metrics_duration ON step_metrics(duration_ms DESC);
CREATE INDEX idx_rank_config_mps ON rank_config(mps_pct);
CREATE INDEX idx_global_step_exp ON global_step_metrics(experiment_id, iter);
CREATE INDEX idx_global_step_straggler ON global_step_metrics(straggler_ratio DESC);
```

## 5. ER Diagram

```
experiments 1──────┬──────* rank_config
                   │              │
                   │              │ 1
                   │              │
                   │              ├──────1 load_metrics        (per-rank init)
                   │              │
                   │              └──────* step_metrics         (per-rank, per-iter)
                   │
                   ├──────1 global_load_metrics                (derived post-training)
                   │
                   └──────* global_step_metrics                (raw from rank 0 + derived)
```

## 6. Example Queries

```sql
-- Find which MPS percentage causes the worst straggler effect
SELECT rc.mps_pct,
       AVG(sm.duration_ms) AS avg_step_ms
FROM   step_metrics sm
JOIN   rank_config rc USING (experiment_id, rank)
GROUP BY rc.mps_pct
ORDER BY avg_step_ms DESC;

-- Per-step communication breakdown for one experiment
SELECT iter,
       AVG(pp_send_cumulative_ms) AS avg_pp_send,
       AVG(pp_recv_cumulative_ms) AS avg_pp_recv,
       AVG(dp_allreduce_ms) AS avg_dp_allreduce,
       MAX(duration_ms) - MIN(duration_ms) AS step_skew_ms
FROM   step_metrics
WHERE  experiment_id = ?
GROUP BY iter
ORDER BY iter;

-- Identify the bottleneck rank per step
SELECT iter, rank, duration_ms,
       pp_send_cumulative_ms, pp_recv_cumulative_ms,
       dp_allreduce_ms
FROM   step_metrics
WHERE  experiment_id = ?
  AND  (experiment_id, iter, duration_ms) IN (
        SELECT experiment_id, iter, MAX(duration_ms)
        FROM step_metrics
        WHERE experiment_id = ?
        GROUP BY iter
       )
ORDER BY iter;

-- Compare schedulers across all completed experiments
SELECT e.scheduler,
       AVG(gs.iter_duration_ms) AS avg_iter_ms,
       AVG(gs.straggler_ratio) AS avg_straggler,
       COUNT(DISTINCT e.experiment_id) AS num_experiments
FROM   global_step_metrics gs
JOIN   experiments e USING (experiment_id)
WHERE  e.status = 'completed'
GROUP BY e.scheduler
ORDER BY avg_iter_ms;
```

## 7. Schema Justification

### Why 6 tables instead of 1 flat table?

| Table | Row count (per experiment) | Rationale |
|-------|---------------------------|-----------|
| `experiments` | 1 | Config is written once, never repeated |
| `rank_config` | 6 | MPS assignment is per-rank, not per-step |
| `load_metrics` | 6 | Init timing is one-shot, not repeated per step |
| `step_metrics` | 6 × 100 = 600 | Bulk data; normalized to avoid repeating config in every row |
| `global_load_metrics` | 1 | Overall model load window — DERIVED post-training |
| `global_step_metrics` | 100 | Raw from rank 0 during training + derived cross-rank post-training |

### Raw vs Derived columns

During training, raw values are recorded (timestamps, byte counts, memory readings) plus CUDA-event cumulative timings (`pp_send_cumulative_ms`, `pp_recv_cumulative_ms`, `forward_compute_ms`, `backward_compute_ms`). Legacy span durations (`pp_send_ms`, `pp_recv_ms`, etc.) and all global aggregates are still computed post-training.

### Why `mps_pct` controls both thread% AND memory?

A single MPS percentage maps to both dimensions: `active_thread_percentage = mps_pct`, `memory_limit_mb = floor(45459 * mps_pct / 100)`. This simplifies the sweep space while maintaining a realistic heterogeneous constraint.

### Why `mps_pct` is in `rank_config` and NOT in `step_metrics`?

MPS percentage is an experiment-level assignment that doesn't change per step. Storing it only in `rank_config` and joining when needed avoids inconsistency. The join `step_metrics JOIN rank_config USING (experiment_id, rank)` is cheap on the composite primary key.

### Why separate `load_metrics` from `step_metrics`?

Load/init happens once per rank, not once per iteration. Mixing it into `step_metrics` would require either a sentinel `iter = -1` row or nullable columns in every step row — both are worse than a clean separate table.

### Why `loss` is nullable in `step_metrics`?

Only the last pipeline stage (pp_stage = 1 in a 2-stage setup) computes loss. Stage-0 ranks have no loss value. `NULL` is semantically correct — it means "not computed here," not "zero."

### Why `straggler_ratio` in `global_step_metrics`?

This is the primary metric differentiating the three schedulers under heterogeneous MPS. Per-step straggler ratio = `MAX(rank_duration) / MIN(rank_duration)`. A perfect scheduler achieves ≈1.0; a naive scheduler with 20% MPS on one node will show >>1.0. Computed post-training from stitched cross-rank data.

## 8. Storage Estimates

| Sweep size | Experiments | step_metrics rows | Approx DB size |
|------------|-------------|-------------------|----------------|
| 3 schedulers × 1 MPS config | 3 | 1,800 | ~1 MB |
| 3 schedulers × 10 MPS configs | 30 | 18,000 | ~5 MB |
| 3 schedulers × 50 MPS configs | 150 | 90,000 | ~25 MB |

SQLite handles this comfortably.

## 9. Quick Start

### Enable metrics collection

Set `ASTEROID_METRICS=1` before deploying. The env var propagates through `generate_manifests.py` → K8s job template → training container.

```bash
export ASTEROID_METRICS=1
./scripts/deploy_asteroid.sh --redeploy
```

Each rank writes to `/tmp/asteroid_metrics/rank_{rank}.jsonl` (hostPath-mounted, persists after pod cleanup).

### Run MPS sweep

```bash
# Sample 20 random experiments from the full 12,288 sweep space
python3 baselines/scripts/run_experiments.py \
    --config baselines/configs/asteroid_default.yaml \
    --num-experiments 20 \
    --db experiments.db \
    --output-dir experiment_data

# Dry run — print experiments without deploying
python3 baselines/scripts/run_experiments.py --num-experiments 10 --dry-run
```

### Stitch JSONL into SQLite

```bash
# Single experiment
python3 baselines/scripts/stitch_metrics.py \
    --experiment-id abc123 \
    --data-dir experiment_data/abc123 \
    --db experiments.db

# All experiments
python3 baselines/scripts/stitch_metrics.py \
    --all --output-dir experiment_data --db experiments.db
```

### Query results

```sql
sqlite3 experiments.db \
  "SELECT e.scheduler, AVG(gs.straggler_ratio) AS avg_straggler
   FROM global_step_metrics gs
   JOIN experiments e USING (experiment_id)
   WHERE e.status = 'completed'
   GROUP BY e.scheduler
   ORDER BY avg_straggler"
```

### File inventory

| File | Purpose |
|------|---------|
| `baselines/utils/metrics.py` | `MetricsCollector` — per-rank JSONL writer |
| `baselines/scripts/run_experiments.py` | Experiment runner (sweep, deploy, collect) |
| `baselines/scripts/stitch_metrics.py` | Post-training JSONL → SQLite stitcher |
| `baselines/docs/PRD_experiment_schema.md` | This file — schema + design docs |
