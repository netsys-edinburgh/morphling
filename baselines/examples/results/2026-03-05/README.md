# Real-Trace Evaluation Report (Reproducible)

Hi Bessus — this report documents **all tests that were executed** in this session,
including commands, metrics, data schemas, and interpretation guidelines.

This file is designed for **human understanding** and **full reproducibility**.

---

## 1) Scope of Experiments

We executed three experiment groups:

1. **Real-trace analytics (offline, trace-to-trace comparison)**
   - Compare `ldpc_trace_with_ctrl.csv` vs `ldpc_trace_without_ctrl.csv`
   - Use `scripts/plot_real_traces.py`
   - Outputs: latency/deadline/violation/inefficiency plots + summary table

2. **Training integration evaluation (runtime benchmark)**
   - Run `scripts/eval_greenctx_training.py` with each real trace
   - Compare step-time and throughput from generated per-step CSVs
   - Outputs: eval CSVs + comparison plots

3. **LLaMA pipeline + GEMM intercept + violation analyzer**
   - Run `baselines/examples/llama_sst_single_gpu.py`
   - Enable greenctx + GEMM interceptor
   - Run `baselines/examples/analyze_violations.py` on generated logs
   - Outputs: gemm log, step boundaries, metrics, violation report

---

## 2) Artifact Layout

All outputs are stored under:

```text
baselines/examples/results/2026-03-05/
├── real_trace_plots/
│   ├── summary.txt
│   ├── deadline_compliance.{png,pdf}
│   ├── decode_latency_cdf.{png,pdf}
│   ├── timeline_decode_and_sm.{png,pdf}
│   ├── workload_vs_latency.{png,pdf}
│   └── violation_inefficiency_events.{png,pdf}
├── eval_with_ctrl/
│   ├── eval_metrics_greenctx.csv
│   ├── eval_iter_time.{png,pdf}
│   ├── eval_loss.{png,pdf}
│   └── eval_throughput.{png,pdf}
├── eval_without_ctrl/
│   └── (same files as eval_with_ctrl)
├── eval_compare/
│   ├── comparison_summary.csv
│   ├── ctrl_vs_noctrl_eval.{png,pdf}
│   └── ctrl_vs_noctrl_step_time_series.{png,pdf}
└── llama_pipeline/
    ├── gemm_log.csv
    ├── violation_report.txt
    ├── violation_summary.csv
    └── output/
        ├── metrics.csv
        ├── step_boundaries.csv
        ├── summary.json
        ├── best.pt
        └── final.pt
```

---

## 3) Data Structures (Schema Reference)

### 3.1 Real trace input schema (`data/ldpc_trace_*.csv`)

Header observed in both files:

```csv
frame,slot,time_slot_sched_ns,time_decode_start_est_ns,time_decode_end_est_ns,time_decode_start_actual_ns,time_decode_end_actual_ns,decode_dur_us,deadline_met,target_sm,profile_idx,sm_count,num_pusch,sum_prb,sum_tbs_bytes,max_mcs
```

Column meaning (key ones):
- `decode_dur_us`: decode latency in microseconds
- `deadline_met`: 1 if decode finished before deadline, else 0
- `target_sm`: intended SM allocation target
- `sm_count`: actual SM count observed/assigned in trace
- `sum_prb`, `sum_tbs_bytes`: traffic/workload magnitude proxies

### 3.2 Eval per-step metrics (`eval_metrics_greenctx.csv`)

```csv
step_idx,wall_time_ms,loss,tokens_per_sec,gpu_mem_mb,sm_count,is_warmup
```

- `wall_time_ms`: end-to-end host-side duration per training step
- `tokens_per_sec`: throughput proxy at step granularity
- `gpu_mem_mb`: allocated memory snapshot
- `sm_count`: SM count recorded by runtime path (0 when backend falls back)
- `is_warmup`: warmup steps marker for filtering stable statistics

### 3.3 LLaMA training metrics (`llama_pipeline/output/metrics.csv`)

```csv
step,loss,accuracy,lr,throughput_samples_per_sec,mem_alloc_mb,mem_reserved_mb,mem_peak_alloc_mb,sm_count
```

Used to monitor optimization behavior and resource usage.

### 3.4 Step boundary log (`llama_pipeline/output/step_boundaries.csv`)

```csv
step,start_ns,end_ns,sm_count
```

- `start_ns/end_ns` use `CLOCK_MONOTONIC` in ns (aligned with GEMM log clock)
- Primary join key for violation analysis (time overlap checks)

### 3.5 GEMM interceptor log (`llama_pipeline/gemm_log.csv`)

```csv
function_name,start_ns,end_ns,duration_ns,m,n,k
```

- One row per intercepted GEMM call
- `m,n,k` are GEMM dimensions
- Time fields are monotonic-ns for overlap analysis

### 3.6 Violation summary (`llama_pipeline/violation_summary.csv`)

```csv
step,next_step,sm_this,sm_next,sm_total,num_violations,max_overshoot_ns,total_overshoot_ns
```

Per-step-boundary aggregation used by the report.

---

## 4) Metrics Used and Why

### 4.1 Latency metrics (`mean`, `p50`, `p95`, `p99`, `max` decode_dur_us)

**Meaning:**
- `mean`: central tendency (average delay)
- `p50`: median experience
- `p95/p99`: tail-latency (SLO-sensitive behavior)
- `max`: worst-case outlier

**Why chosen:**
- Control strategies often improve tails more than means.
- `p95/p99` are necessary for deadline-critical telecom workloads.

### 4.2 Deadline miss rate (%)

**Meaning:** percent of samples where `deadline_met==0`.

**Why chosen:**
- Directly maps to correctness/service objective.
- More actionable than latency alone in deadline systems.

### 4.3 Violation count / inefficiency count

**Meaning:** events detected by adapter logic:
- violation: policy/scheduling conflict events
- inefficiency: underutilization/waste events

**Why chosen:**
- Operationally explains *why* latency/deadline changes happen.
- Connects outcome metrics to control behavior.

### 4.4 Step wall time (`wall_time_ms`) and throughput (`tokens_per_sec`)

**Meaning:** runtime performance in integrated training loop.

**Why chosen:**
- `wall_time_ms` captures end-to-end per-step cost.
- `tokens_per_sec` provides normalized speed metric for training.

### 4.5 Violation-time metrics (from analyzer)

**Meaning:**
- slot violation %: fraction of step boundaries with violations
- time with violation %: fraction of training wall-time in violation state

**Why chosen:**
- Converts raw event counts into interpretable severity and duty cycle.

---

## 5) Executed Results

## 5.1 Real-trace analytics (`real_trace_plots/summary.txt`)

| Metric | with_ctrl | without_ctrl | improvement_% |
|---|---:|---:|---:|
| mean decode_dur_us | 260.523 | 306.185 | +14.91 |
| p50 decode_dur_us | 213.000 | 260.000 | +18.08 |
| p95 decode_dur_us | 602.000 | 625.000 | +3.68 |
| p99 decode_dur_us | 697.000 | 715.000 | +2.52 |
| max decode_dur_us | 1276.000 | 1015.000 | -25.71 |
| deadline miss % | 9.868 | 11.973 | +17.58 |
| violation count | 1683 | 3285 | +48.77 |
| inefficiency count | 1793 | 3795 | +52.75 |

Interpretation:
- Control improves central and tail latency, deadline misses, and event counts.
- Max latency worsens in this run (outlier sensitivity); report both max and quantiles.

## 5.2 Training integration eval (`eval_compare/comparison_summary.csv`)

Post-warmup (25 steps each):
- with_ctrl mean step time: **29.6656 ms**
- without_ctrl mean step time: **28.5806 ms**
- with_ctrl mean throughput: **8630.32 tok/s**
- without_ctrl mean throughput: **8958.60 tok/s**
- derived deltas:
  - `iter_time_improvement_pct = -3.7964%`
  - `throughput_improvement_pct = -3.6644%`

Important runtime note:
- During both eval runs, runtime logs show **TorchNative green-context creation failed** due driver capability and fallback to **OffBackend** (`supported=False`).
- Therefore these integration numbers compare trace-conditioned behavior under the same fallback backend, not hard green-context isolation.

## 5.3 LLaMA + GEMM + analyzer (`llama_pipeline/`)

From `violation_report.txt`:
- GEMM log entries: **5332**
- Training steps: **5**
- Slots with violations: **0/4 (0.00%)**
- Time with violations: **0ns / 727.22ms (0.0000%)**
- Violating GEMM calls: **0/5332 (0.00%)**

From `summary.json`:
- `greenctx_enabled = true`
- `greenctx_supported = true`
- `max_sm_count = 48`

From `step_boundaries.csv`:
- step-level `sm_count` recorded as 32 across 5 steps.

---

## 6) Reproduction Commands

Run from repo root (`/home/xly/DeviceEmulator`):

### 6.1 Real trace plots

```bash
docker run --rm -v "$(pwd):/workspace" device-emulator:latest bash -lc '
mkdir -p /workspace/baselines/examples/results/2026-03-05/real_trace_plots &&
python3 -m pip install -q matplotlib pandas numpy &&
python3 /workspace/scripts/plot_real_traces.py   --with-ctrl /workspace/data/ldpc_trace_with_ctrl.csv   --without-ctrl /workspace/data/ldpc_trace_without_ctrl.csv   --output-dir /workspace/baselines/examples/results/2026-03-05/real_trace_plots   --total-sms 48   > /workspace/baselines/examples/results/2026-03-05/real_trace_plots/summary.txt'
```

### 6.2 Integration eval: with control trace

```bash
docker run --rm --gpus "device=0" -v "$(pwd):/workspace" device-emulator:latest bash -lc '
mkdir -p /workspace/baselines/examples/results/2026-03-05/eval_with_ctrl &&
python3 -m pip install -q matplotlib pandas numpy datasets &&
python3 /workspace/scripts/eval_greenctx_training.py   --trace-path /workspace/data/ldpc_trace_with_ctrl.csv   --output-dir /workspace/baselines/examples/results/2026-03-05/eval_with_ctrl   --num-steps 30 --batch-size 2 --seq-len 128 --skip-baseline'
```

### 6.3 Integration eval: without control trace

```bash
docker run --rm --gpus "device=0" -v "$(pwd):/workspace" device-emulator:latest bash -lc '
mkdir -p /workspace/baselines/examples/results/2026-03-05/eval_without_ctrl &&
python3 -m pip install -q matplotlib pandas numpy datasets &&
python3 /workspace/scripts/eval_greenctx_training.py   --trace-path /workspace/data/ldpc_trace_without_ctrl.csv   --output-dir /workspace/baselines/examples/results/2026-03-05/eval_without_ctrl   --num-steps 30 --batch-size 2 --seq-len 128 --skip-baseline'
```

### 6.4 LLaMA pipeline + GEMM + violation analysis

```bash
docker run --rm --gpus "device=0" -v "$(pwd):/workspace" device-emulator:latest bash -lc '
mkdir -p /workspace/baselines/examples/results/2026-03-05/llama_pipeline &&
python3 -m pip install -q datasets &&
make -C /workspace/baselines/examples/gemm_intercept >/dev/null &&
LD_PRELOAD=/workspace/baselines/examples/gemm_intercept/libgemm_intercept.so:/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12 GEMM_LOG_PATH=/workspace/baselines/examples/results/2026-03-05/llama_pipeline/gemm_log.csv python3 /workspace/baselines/examples/llama_sst_single_gpu.py   --max-iters 5 --batch-size 8 --eval-interval 10 --log-interval 1   --cuda-id 0 --greenctx-enabled --greenctx-backend auto   --greenctx-trace /workspace/baselines/examples/sample_greenctx_trace.csv   --max-sm-count 48   --gemm-log /workspace/baselines/examples/results/2026-03-05/llama_pipeline/gemm_log.csv   --output-dir /workspace/baselines/examples/results/2026-03-05/llama_pipeline/output &&
python3 /workspace/baselines/examples/analyze_violations.py   --gemm-log /workspace/baselines/examples/results/2026-03-05/llama_pipeline/gemm_log.csv   --step-log /workspace/baselines/examples/results/2026-03-05/llama_pipeline/output/step_boundaries.csv   --trace /workspace/baselines/examples/sample_greenctx_trace.csv   --max-sm-count 48   --output /workspace/baselines/examples/results/2026-03-05/llama_pipeline/violation_report.txt   --output-csv /workspace/baselines/examples/results/2026-03-05/llama_pipeline/violation_summary.csv'
```

---

## 7) Reproducibility Checklist

- [x] Fixed output root with timestamped directory
- [x] Raw CSV logs preserved (trace/eval/step/gemm/violation)
- [x] Plot artifacts saved as PNG and PDF
- [x] Explicit command lines documented
- [x] Metric definitions and rationale documented
- [x] Data schemas documented
- [x] Runtime caveat (OffBackend fallback) documented

---

## 8) Recommended Reading Order for Humans

1. `real_trace_plots/summary.txt` (topline outcomes)
2. `real_trace_plots/decode_latency_cdf.png` (distributional view)
3. `eval_compare/comparison_summary.csv` (integration runtime numbers)
4. `llama_pipeline/violation_report.txt` (GEMM overlap/violation status)
5. This README (methodology, schema, and reproducibility details)

