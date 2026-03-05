# LLaMA + SST-2 Single-GPU Training with SM Partition Violation Analysis

This guide explains how to train a small LLaMA model on the SST-2 sentiment classification dataset with:
- **CUDA Green Context SM partitioning** — dynamic SM allocation per training step
- **GEMM timing interception** — nanosecond-precision measurement of cuBLAS calls
- **Violation analysis** — detect when GEMM operations spill past step boundaries while SM pressure exceeds thresholds

---

## Quick Start (TL;DR)

```bash
# 1. Build Docker image
docker build -t device-emulator:latest .

# 2. Run full pipeline: train + intercept + analyze
docker run --rm --gpus all -v $(pwd):/workspace device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh both --max-iters 100

# 3. Run violation analysis on results
docker run --rm --gpus all -v $(pwd):/workspace device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh analyze
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  SST-2 Data  │───>│ LLaMA Model  │───>│ Green Context│      │
│  │  (HF datasets)│   │ (512d, 6L)   │    │ Controller   │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                    ┌─────────────────────────────┘              │
│                    │                                           │
│                    ▼                                           │
│  ┌────────────────────────────────┐                           │
│  │  step_scope(step_idx)          │                           │
│  │  ├── activate_for_step()       │                           │
│  │  ├── bundle.comp stream        │                           │
│  │  │   └── forward/backward      │                           │
│  │  └── deactivate()              │                           │
│  └────────────────────────────────┘                           │
│                    │                                           │
│                    ▼                                           │
│  ┌────────────────────────────────┐                           │
│  │  GEMM Intercept (LD_PRELOAD)   │                           │
│  │  ├── cublasGemmEx              │                           │
│  │  ├── cublasSgemm_v2            │                           │
│  │  ├── cublasLtMatmul            │                           │
│  │  └── Log: start_ns, end_ns     │                           │
│  └────────────────────────────────┘                           │
│                    │                                           │
│                    ▼                                           │
│  ┌────────────────────────────────┐                           │
│  │  Output Files                  │                           │
│  │  ├── step_boundaries.csv       │                           │
│  │  ├── gemm_log.csv              │                           │
│  │  ├── metrics.csv               │                           │
│  │  └── summary.json              │                           │
│  └────────────────────────────────┘                           │
│                    │                                           │
│                    ▼                                           │
│  ┌────────────────────────────────┐                           │
│  │  Violation Analysis            │                           │
│  │  ├── Parse GEMM logs           │                           │
│  │  ├── Parse step boundaries     │                           │
│  │  ├── Detect overlaps           │                           │
│  │  └── Report violations         │                           │
│  └────────────────────────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `baselines/examples/llama_sst_single_gpu.py` | Main training script |
| `baselines/examples/gemm_intercept/gemm_intercept.c` | C GEMM interceptor (LD_PRELOAD) |
| `baselines/examples/gemm_intercept/Makefile` | Build system for interceptor |
| `baselines/examples/gemm_intercept/README.md` | GEMM intercept documentation |
| `baselines/examples/analyze_violations.py` | Post-training violation analyzer |
| `baselines/examples/sample_greenctx_trace.csv` | Sample SM allocation trace |
| `baselines/examples/run_llama_sst.sh` | Launcher script for all modes |

---

## Prerequisites

### System Requirements

- NVIDIA GPU with CUDA 12.x support
- Docker with NVIDIA Container Toolkit (`--gpus` support)
- ~10GB GPU memory (for the small LLaMA config: 512d, 6 layers, 8 heads)

### Building

All dependencies are in the Docker image:

```bash
docker build -t device-emulator:latest .
```

---

## Usage

### Mode 1: Basic Training (No Interception)

Train without green context or GEMM interception:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh basic \
  --max-iters 500 \
  --batch-size 32 \
  --output-dir ./output/basic
```

### Mode 2: Green Context SM Partitioning

Train with dynamic SM allocation per step:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh greenctx \
  --max-iters 500 \
  --batch-size 32 \
  --greenctx-trace baselines/examples/sample_greenctx_trace.csv \
  --max-sm-count 48 \
  --output-dir ./output/greenctx
```

**What happens:**
- At step 0: allocates 32 SMs
- At step 50: switches to 16 SMs
- At step 100: switches to 48 SMs
- At step 150: switches back to 32 SMs

### Mode 3: GEMM Timing Interception

Train with nanosecond-precision GEMM logging:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh gemm \
  --max-iters 500 \
  --batch-size 32 \
  --output-dir ./output/gemm
```

**Note:** The launcher automatically sets up `LD_PRELOAD` with the correct cuBLAS library chain.

### Mode 4: Both (Green Context + GEMM Interception)

Full pipeline with SM partitioning + GEMM timing:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh both \
  --max-iters 500 \
  --batch-size 32 \
  --greenctx-trace baselines/examples/sample_greenctx_trace.csv \
  --max-sm-count 48 \
  --output-dir ./output/both
```

### Mode 5: Violation Analysis

Analyze SM partition violations from previous training run:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  bash /app/baselines/examples/run_llama_sst.sh analyze
```

Or run directly:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  device-emulator:latest \
  python3 /app/baselines/examples/analyze_violations.py \
  --gemm-log ./output/both/gemm_log.csv \
  --step-log ./output/both/step_boundaries.csv \
  --trace baselines/examples/sample_greenctx_trace.csv \
  --max-sm-count 48 \
  --output ./output/both/violation_report.txt \
  --output-csv ./output/both/violation_summary.csv \
  --verbose
```

---

## Understanding the Output

### Training Outputs

After training, the output directory contains:

```
./output/
├── best.pt                 # Best model checkpoint (by val accuracy)
├── final.pt                # Final model checkpoint
├── metrics.csv             # Per-step training metrics
├── step_boundaries.csv     # Step timing + SM allocation (NEW)
├── summary.json            # Training summary
└── gemm_log.csv            # GEMM timing logs (if interception enabled)
```

#### step_boundaries.csv

```csv
step,start_ns,end_ns,sm_count
0,1234567890000,1234567990000,32
1,1234567990001,1234568090000,32
2,1234568090001,1234568190000,32
...
```

- `step`: Training step index
- `start_ns`: Step start time (CLOCK_MONOTONIC, nanoseconds)
- `end_ns`: Step end time
- `sm_count`: SM allocation for this step (clamped to `--max-sm-count`)

#### gemm_log.csv

```csv
function_name,start_ns,end_ns,duration_ns,m,n,k
cublasGemmEx,1234567890100,1234567890150,50,512,1024,512
cublasGemmEx,1234567890200,1234567890250,50,512,512,512
...
```

- `function_name`: cuBLAS function (cublasGemmEx, cublasLtMatmul, etc.)
- `start_ns/end_ns`: GEMM execution boundaries
- `duration_ns`: `end_ns - start_ns`
- `m,n,k`: GEMM dimensions

### Violation Analysis Report

Example output:

```
=== SM Partition Violation Analysis ===

Configuration:
  Max SM count threshold: 48
  GEMM log entries: 4817
  Training steps: 500
  Total wall time: 45.67s

Summary:
  Slots with violations: 12/499 (2.40%)
  Time with violations: 1.23ms / 45.67s (0.003%)
  Total violating GEMM calls: 47/4817 (0.98%)

Per-step violations (top 10 by overshoot):
  Step 49->50: 8 violations, max_overshoot=234us, sm_pressure=32+16=48 > 48
  Step 99->100: 5 violations, max_overshoot=189us, sm_pressure=16+48=64 > 48
  ...

Detailed violations (first 20):
  step=49 gemm=cublasGemmEx m=512 n=1024 k=512 overshoot=234123ns sm=32+16=48
  ...
```

#### Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Slots with violations** | % of step boundaries where at least one GEMM spilled into next step AND combined SM pressure exceeded threshold |
| **Time with violations** | % of total training time spent in violation state (merged overlapping intervals) |
| **Violating GEMM calls** | % of all GEMM operations that caused violations |
| **SM pressure** | Sum of SM counts at adjacent steps (e.g., 32+16=48) |

---

## How Violation Detection Works

A **violation** occurs when BOTH conditions are true:

1. **Temporal spill**: A GEMM operation's `end_ns` exceeds the `start_ns` of the NEXT step
   ```
   Step i:     [=========GEMM=========>]
   Step i+1:            [=====>]
                          ^^^^
                          spill here
   ```

2. **SM pressure**: `sm_count[i] + sm_count[i+1] > max_sm_count`
   ```
   Example: sm_count[49]=32, sm_count[50]=16, max_sm=48
   32 + 16 = 48 → NOT a violation (equal is OK)
   
   Example: sm_count[99]=16, sm_count[100]=48, max_sm=48
   16 + 48 = 64 > 48 → POTENTIAL violation (if GEMM spills)
   ```

### Why Both Conditions?

- **SM partitioning alone** (via CUDA green contexts) provides **hard isolation** — kernels physically cannot run on non-allocated SMs
- However, if a GEMM operation starts in step `i` and finishes in step `i+1`, the GPU would temporarily have both step workloads active
- If the combined SM allocation exceeds the physical limit (e.g., 64 SMs on an A100), the hardware would need to timeslice, degrading performance

---

## Configuration

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 32 | Training batch size |
| `--max-iters` | 500 | Number of training steps |
| `--lr` | 3e-4 | Learning rate |
| `--cuda-id` | 0 | GPU device ID |
| `--seed` | 42 | Random seed |
| `--max-sm-count` | 48 | Max SM threshold for violation detection |
| `--greenctx-enabled` | False | Enable green context SM partitioning |
| `--greenctx-backend` | auto | Backend: auto, cpp, torch_native, off |
| `--greenctx-trace` | None | Path to SM allocation trace CSV |
| `--gemm-log` | None | Path for GEMM log output |
| `--output-dir` | ./output | Output directory |

### SM Allocation Trace Format

Create a CSV file to control SM allocation per step:

```csv
# clock_mode=step
step,num_sms
0,32
50,16
100,48
150,32
200,64
```

- Lines starting with `#` are comments/directives
- `clock_mode=step` means the first column is step index
- The SM count is applied via step-function (last entry where `step <= current_step`)

---

## Troubleshooting

### "Green context import failed"

The green context backend may not be available on older GPUs or CUDA versions. The script gracefully falls back to default streams. Check logs:

```
Green context backend unsupported; using default streams
```

### GEMM log is empty or missing

The LD_PRELOAD chain must include the real cuBLAS libraries:

```bash
# Correct order: interceptor FIRST, then real libs
LD_PRELOAD=libgemm_intercept.so:libcublas.so.12:libcublasLt.so.12
```

The launcher script handles this automatically.

### Out of memory

Reduce batch size or model size:

```bash
--batch-size 16  # instead of 32
```

The small LLaMA config (512d, 6 layers) uses ~2-3GB GPU memory.

---

## Advanced Usage

### Custom SM Traces

Create aggressive SM variations to stress-test violation detection:

```csv
# clock_mode=step
step,num_sms
0,16
10,48
20,8
30,56
40,16
```

### Analyzing Specific Steps

Filter the analysis to specific step ranges:

```bash
python3 analyze_violations.py \
  --gemm-log gemm_log.csv \
  --step-log step_boundaries.csv \
  --max-sm-count 48 \
  --verbose 2>&1 | grep "step=4[0-9]"
```

### Comparing Different Configurations

Run multiple experiments and compare violation rates:

```bash
for max_sm in 32 40 48 56; do
  docker run --rm --gpus all device-emulator:latest \
    bash /app/baselines/examples/run_llama_sst.sh both \
    --max-iters 200 --max-sm-count $max_sm \
    --output-dir ./output/sm_${max_sm}
done

# Compare reports
for dir in ./output/sm_*; do
  echo "=== $dir ==="
  head -20 $dir/violation_report.txt
done
```

---

## References

- CUDA Green Contexts: [NVIDIA CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)
- cuBLAS GEMM: [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- SST-2 Dataset: [GLUE Benchmark](https://gluebenchmark.com/tasks)
- LLaMA Architecture: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

---

## License

Apache 2.0 — see project root LICENSE file.
