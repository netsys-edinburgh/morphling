# PRD: Experiment Data Schema for Scheduler Evaluation

## 1. Context

This schema supports scheduler evaluation across heterogeneous per-rank MPS settings for:
- `asteroid`
- `confident`
- `dtfm`

The experiment system records:
- per-rank runtime metrics (load + per-iteration)
- stitched cross-rank global iteration metrics
- per-microbatch op timelines and derived non-overlap compute/communication metrics

Important: MPS values are now treated as configurable runtime inputs. They are not hardcoded to a fixed enum in the schema; `rank_config.mps_pct` uses a bounded numeric range.

## 2. Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Normalized relational tables** | Avoid repeated config values and make cross-experiment analysis easy. |
| **Raw-first ingestion** | Runtime logs raw timestamps/event offsets; stitched tables hold derived durations and aggregates. |
| **Rank-local logging** | Each rank writes JSONL locally; no cross-rank sync traffic for metrics collection. |
| **Post-run stitching** | Derived metrics (durations, straggler, global totals/max) are computed centrally from collected JSONL. |
| **Compatibility with legacy fields** | Legacy envelope span timings are retained while newer cumulative and microbatch metrics are added. |

## 3. Data Collection Flow

### During training
- Each rank writes `/tmp/asteroid_metrics/rank_{rank}.jsonl`.
- Rank 0 also writes `global_step_raw` once per iteration.
- Runtime emits:
  - load records
  - step records
  - micro-op records (`type: micro_op`) with CUDA-event offsets

### Post-training
1. Runner copies rank JSONL files into `experiment_data/<experiment_id>/`.
2. Stitcher parses JSONL, inserts node/global iteration rows, and computes derived tables.
3. For microbatch metrics, stitcher:
   - stores raw ops in `microbatch_op_metrics`
   - derives per-rank non-overlap rows in `microbatch_metrics`
   - aggregates per-iter/per-microbatch global rows in `global_microbatch_metrics`

## 4. JSONL Record Types

```json
{"type":"load","rank":0,"model_load_start_ts":"...","model_load_end_ts":"...","memory_after_load_mb":1850.5}
{"type":"step","rank":0,"iter":5,"start_ts":"...","end_ts":"...","forward_start_ts":"...","forward_end_ts":"...","pp_send_cumulative_ms":12.3,"forward_compute_ms":45.6}
{"type":"global_step_raw","iter":5,"loss":0.89,"lr":6e-06,"iter_start_ts":"...","iter_end_ts":"..."}
{"type":"micro_op","rank":0,"iter":5,"microbatch":2,"op_seq":17,"op_kind":"fwd_pp_send","stream":"send","start_offset_ms":123.4,"end_offset_ms":124.9,"duration_ms":1.5,"nbytes":8388608,"peer_rank":3}
```

`micro_op.op_kind` values currently emitted:
- `fwd_compute`
- `bwd_compute`
- `fwd_pp_send`
- `fwd_pp_recv`
- `bwd_pp_send`
- `bwd_pp_recv`

## 5. Schema (SQLite)

```sql
-- TABLE 1: experiments
CREATE TABLE experiments (
    experiment_id       TEXT PRIMARY KEY,
    scheduler           TEXT NOT NULL
                        CHECK (scheduler IN ('dtfm', 'confident', 'asteroid')),
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    failure_reason      TEXT,
    git_commit          TEXT,
    docker_image        TEXT,

    model_name          TEXT NOT NULL,
    model_type          TEXT NOT NULL,
    num_layers          INTEGER NOT NULL,
    embedding_dim       INTEGER NOT NULL,
    num_heads           INTEGER NOT NULL,
    d_ff                INTEGER NOT NULL,
    vocab_size          INTEGER NOT NULL,
    max_seq_len         INTEGER NOT NULL,

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

    schedule_type       TEXT NOT NULL,
    pp_size             INTEGER NOT NULL,
    dp_size             INTEGER NOT NULL,
    world_size          INTEGER NOT NULL,
    comm_backend        TEXT NOT NULL
);

-- TABLE 2: rank_config
CREATE TABLE rank_config (
    experiment_id       TEXT NOT NULL REFERENCES experiments(experiment_id),
    rank                INTEGER NOT NULL,
    hostname            TEXT NOT NULL,
    ip                  TEXT NOT NULL,
    gpu_id              INTEGER NOT NULL DEFAULT 0,
    pp_stage            INTEGER NOT NULL,
    dp_group            INTEGER NOT NULL,
    mps_pct             INTEGER NOT NULL
                        CHECK (mps_pct >= 1 AND mps_pct <= 100),
    memory_limit_mb     INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, rank)
);

-- TABLE 3: load_metrics
CREATE TABLE load_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    model_load_start_ts     TIMESTAMP,
    model_load_end_ts       TIMESTAMP,
    model_load_duration_ms  REAL,
    nccl_init_start_ts      TIMESTAMP,
    nccl_init_end_ts        TIMESTAMP,
    nccl_init_duration_ms   REAL,
    memory_after_load_mb    REAL,
    PRIMARY KEY (experiment_id, rank),
    FOREIGN KEY (experiment_id, rank)
        REFERENCES rank_config(experiment_id, rank)
);

-- TABLE 4: step_metrics
CREATE TABLE step_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    iter                    INTEGER NOT NULL,

    start_ts                TIMESTAMP,
    end_ts                  TIMESTAMP,
    duration_ms             REAL,

    forward_start_ts        TIMESTAMP,
    forward_end_ts          TIMESTAMP,
    forward_ms              REAL,
    backward_start_ts       TIMESTAMP,
    backward_end_ts         TIMESTAMP,
    backward_ms             REAL,
    optimizer_start_ts      TIMESTAMP,
    optimizer_end_ts        TIMESTAMP,
    optimizer_step_ms       REAL,

    pp_send_start_ts        TIMESTAMP,
    pp_send_end_ts          TIMESTAMP,
    pp_send_ms              REAL,
    pp_recv_start_ts        TIMESTAMP,
    pp_recv_end_ts          TIMESTAMP,
    pp_recv_ms              REAL,
    pp_send_cumulative_ms   REAL,
    pp_recv_cumulative_ms   REAL,
    forward_compute_ms      REAL,
    backward_compute_ms     REAL,
    pp_send_count           INTEGER,
    pp_recv_count           INTEGER,
    pp_send_bytes           BIGINT,
    pp_recv_bytes           BIGINT,

    dp_allreduce_start_ts   TIMESTAMP,
    dp_allreduce_end_ts     TIMESTAMP,
    dp_allreduce_ms         REAL,
    dp_allreduce_bytes      BIGINT,

    loss                    REAL,
    lr                      REAL,
    memory_allocated_mb     REAL,
    memory_reserved_mb      REAL,
    memory_peak_mb          REAL,

    PRIMARY KEY (experiment_id, rank, iter),
    FOREIGN KEY (experiment_id, rank)
        REFERENCES rank_config(experiment_id, rank)
);

-- TABLE 5: microbatch_op_metrics (raw micro-op timeline)
CREATE TABLE microbatch_op_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    iter                    INTEGER NOT NULL,
    microbatch              INTEGER NOT NULL,
    op_seq                  INTEGER NOT NULL,
    op_kind                 TEXT NOT NULL,
    stream                  TEXT NOT NULL,
    start_offset_ms         REAL,
    end_offset_ms           REAL,
    duration_ms             REAL,
    nbytes                  BIGINT,
    peer_rank               INTEGER,
    PRIMARY KEY (experiment_id, rank, iter, microbatch, op_seq),
    FOREIGN KEY (experiment_id, rank, iter)
        REFERENCES step_metrics(experiment_id, rank, iter)
);

-- TABLE 6: microbatch_metrics (derived per-rank/per-microbatch)
CREATE TABLE microbatch_metrics (
    experiment_id           TEXT NOT NULL,
    rank                    INTEGER NOT NULL,
    iter                    INTEGER NOT NULL,
    microbatch              INTEGER NOT NULL,
    compute_nonoverlap_ms   REAL,
    comm_nonoverlap_ms      REAL,
    total_union_ms          REAL,
    compute_sum_ms          REAL,
    comm_sum_ms             REAL,
    total_sum_ms            REAL,
    PRIMARY KEY (experiment_id, rank, iter, microbatch),
    FOREIGN KEY (experiment_id, rank, iter)
        REFERENCES step_metrics(experiment_id, rank, iter)
);

-- TABLE 7: global_load_metrics
CREATE TABLE global_load_metrics (
    experiment_id               TEXT PRIMARY KEY
        REFERENCES experiments(experiment_id),
    overall_load_start_ts       TIMESTAMP,
    overall_load_end_ts         TIMESTAMP,
    overall_load_duration_ms    REAL,
    slowest_rank                INTEGER,
    slowest_rank_load_ms        REAL,
    fastest_rank                INTEGER,
    fastest_rank_load_ms        REAL,
    overall_nccl_init_ms        REAL,
    total_memory_after_load_mb  REAL
);

-- TABLE 8: global_step_metrics
CREATE TABLE global_step_metrics (
    experiment_id                   TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    iter                            INTEGER NOT NULL,
    iter_start_ts                   TIMESTAMP,
    iter_end_ts                     TIMESTAMP,
    loss                            REAL,
    lr                              REAL,
    iter_duration_ms                REAL,
    total_forward_ms                REAL,
    total_backward_ms               REAL,
    total_optimizer_ms              REAL,
    total_pp_send_ms                REAL,
    total_pp_recv_ms                REAL,
    total_pp_send_cumulative_ms     REAL,
    total_pp_recv_cumulative_ms     REAL,
    total_pp_bytes                  BIGINT,
    max_pp_send_ms                  REAL,
    max_pp_recv_ms                  REAL,
    max_pp_send_cumulative_ms       REAL,
    max_pp_recv_cumulative_ms       REAL,
    total_forward_compute_ms        REAL,
    total_backward_compute_ms       REAL,
    total_dp_allreduce_ms           REAL,
    max_dp_allreduce_ms             REAL,
    dp_allreduce_bytes              BIGINT,
    slowest_rank                    INTEGER,
    fastest_rank                    INTEGER,
    straggler_ratio                 REAL,
    PRIMARY KEY (experiment_id, iter)
);

-- TABLE 9: global_microbatch_metrics (cross-rank aggregation)
CREATE TABLE global_microbatch_metrics (
    experiment_id                   TEXT NOT NULL
        REFERENCES experiments(experiment_id),
    iter                            INTEGER NOT NULL,
    microbatch                      INTEGER NOT NULL,
    total_compute_nonoverlap_ms     REAL,
    total_comm_nonoverlap_ms        REAL,
    total_union_ms                  REAL,
    total_compute_sum_ms            REAL,
    total_comm_sum_ms               REAL,
    total_sum_ms                    REAL,
    max_compute_nonoverlap_ms       REAL,
    max_comm_nonoverlap_ms          REAL,
    max_total_union_ms              REAL,
    slowest_rank                    INTEGER,
    fastest_rank                    INTEGER,
    straggler_ratio                 REAL,
    PRIMARY KEY (experiment_id, iter, microbatch)
);

-- INDEXES
CREATE INDEX idx_experiments_scheduler
    ON experiments(scheduler);
CREATE INDEX idx_experiments_status
    ON experiments(status);
CREATE INDEX idx_step_metrics_exp_rank
    ON step_metrics(experiment_id, rank);
CREATE INDEX idx_step_metrics_duration
    ON step_metrics(duration_ms DESC);
CREATE INDEX idx_micro_op_exp_iter
    ON microbatch_op_metrics(experiment_id, iter);
CREATE INDEX idx_micro_op_rank_iter
    ON microbatch_op_metrics(experiment_id, rank, iter);
CREATE INDEX idx_micro_metrics_exp_iter
    ON microbatch_metrics(experiment_id, iter);
CREATE INDEX idx_micro_metrics_rank_iter
    ON microbatch_metrics(experiment_id, rank, iter);
CREATE INDEX idx_rank_config_mps
    ON rank_config(mps_pct);
CREATE INDEX idx_global_step_exp
    ON global_step_metrics(experiment_id, iter);
CREATE INDEX idx_global_step_straggler
    ON global_step_metrics(straggler_ratio DESC);
CREATE INDEX idx_global_micro_exp_iter
    ON global_microbatch_metrics(experiment_id, iter);
```

## 6. ER Diagram

```
experiments 1──────┬──────* rank_config
                   │              │
                   │              ├──────1 load_metrics
                   │              └──────* step_metrics
                   │                        ├──────* microbatch_op_metrics
                   │                        └──────* microbatch_metrics
                   │
                   ├──────1 global_load_metrics
                   ├──────* global_step_metrics
                   └──────* global_microbatch_metrics
```

## 7. Metric Semantics (Important)

### Legacy envelope spans (`*_ms` from timestamp boundaries)
Fields such as `forward_ms`, `pp_send_ms`, and `pp_recv_ms` are envelope spans between first start and last end markers for that phase in an iteration. They can overlap and should not be summed as disjoint phase time.

### Cumulative CUDA-event fields (iteration-level)
`pp_send_cumulative_ms`, `pp_recv_cumulative_ms`, `forward_compute_ms`, and `backward_compute_ms` are event-based cumulative sums across recorded ops in an iteration.

### Microbatch non-overlap fields
Per `(experiment_id, rank, iter, microbatch)`, stitcher computes interval unions:
- `compute_nonoverlap_ms`: union of compute intervals
- `comm_nonoverlap_ms`: union of communication intervals
- `total_union_ms`: union of all intervals

Also tracked for diagnostics:
- `compute_sum_ms`, `comm_sum_ms`, `total_sum_ms` (plain sums, overlap-allowing)

Global microbatch rows aggregate per-rank derived rows using totals/max values and straggler stats (no cross-rank timeline union).

## 8. Example Queries

```sql
-- 1) Per-rank microbatch non-overlap profile
SELECT iter, microbatch,
       compute_nonoverlap_ms,
       comm_nonoverlap_ms,
       total_union_ms,
       compute_sum_ms,
       comm_sum_ms
FROM microbatch_metrics
WHERE experiment_id = ? AND rank = ?
ORDER BY iter, microbatch;

-- 2) Global microbatch straggler view
SELECT iter, microbatch,
       total_union_ms,
       max_total_union_ms,
       slowest_rank,
       fastest_rank,
       straggler_ratio
FROM global_microbatch_metrics
WHERE experiment_id = ?
ORDER BY iter, microbatch;

-- 3) Compute-vs-comm diagnostics (iteration level)
SELECT iter,
       total_forward_compute_ms,
       total_backward_compute_ms,
       total_pp_send_cumulative_ms,
       total_pp_recv_cumulative_ms,
       total_dp_allreduce_ms
FROM global_step_metrics
WHERE experiment_id = ?
ORDER BY iter;

-- 4) Scheduler comparison across completed experiments
SELECT e.scheduler,
       AVG(gs.iter_duration_ms) AS avg_iter_ms,
       AVG(gs.straggler_ratio) AS avg_straggler,
       COUNT(DISTINCT e.experiment_id) AS num_experiments
FROM global_step_metrics gs
JOIN experiments e USING (experiment_id)
WHERE e.status = 'completed'
GROUP BY e.scheduler
ORDER BY avg_iter_ms;
```

## 9. Storage Notes

Storage growth is dominated by:
- `step_metrics`: O(ranks × iters)
- `microbatch_op_metrics`: O(ranks × iters × microbatches × ops_per_microbatch)
- `microbatch_metrics`: O(ranks × iters × microbatches)
- `global_microbatch_metrics`: O(iters × microbatches)

In practice, microbatch timeline capture can exceed iteration-table size, so retention/archival strategy should account for the micro-op table volume.

## 10. Quick Start

### Enable metrics collection and run experiments

```bash
export ASTEROID_METRICS=1
python3 baselines/scripts/run_experiments.py \
  --config baselines/configs/asteroid_default.yaml \
  --db baselines/experiments.db \
  --output-dir baselines/experiment_data
```

### Stitch JSONL into SQLite

```bash
python3 baselines/scripts/stitch_metrics.py \
  --all \
  --output-dir baselines/experiment_data \
  --db baselines/experiments.db
```

### Export Markdown tables (including microbatch sections)

```bash
python3 baselines/scripts/export_metrics_tables_md.py \
  --db baselines/experiments.db \
  --output baselines/experiment_data/reports/latest_metrics_tables.md \
  --schedulers asteroid,confident,dtfm \
  --include-microbatch
```

## 11. Source of Truth

The operational schema and stitch behavior are defined in:
- `baselines/scripts/run_experiments.py`
- `baselines/scripts/stitch_metrics.py`

This PRD is documentation-only and must track those files when schema changes.
