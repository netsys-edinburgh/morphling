# DeviceEmulator

DeviceEmulator (Morphling) is a device and network emulator for distributed
inference workflows. It provides a Python runtime, C++ backend, and scripts to
run virtual or physical device configurations.

## Installation

```bash
# Use conda or virtualenv
pip install -e .

# If using conda, some Torch dependencies may require libxslt
conda install -c conda-forge libxslt

# If you have CUDA installed, make it discoverable for Torch
export CUDA_HOME=/usr/local/cuda

# If Torch import/build errors occur, try rebuilding without isolation
pip install --no-build-isolation .
```

## Quick Start

```bash
morphling_cmd save --model "facebook/opt-125m" --output <path to model checkpoint>
morphling_emulator --ckpt_path <path to model checkpoint>
```

## Usage

### Virtual Device Usage

You can enable low-memory emulator transport mode by setting
`transport_mode=emulator` in both `config/proxy/cli.ini` and
`config/proxy/svr.ini`.

```bash
# Start Redis (stop existing container if needed)
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

cd scripts
SPDLOG_LEVEL=debug python run_devices.py \
    --num_devices 4 \
    --model_name facebook/opt-125m \
    --backend proxy \
    --seq_length 128 \
    --batch_size 1 \
    --cfg ../config/proxy/svr.ini
```

### Physical Device Usage

```bash
#!/usr/bin/env bash
set -e

# 1) Remove any existing redis or morphling containers
REDIS_CONTAINERS=$(docker ps -aq -f name=redis)
MORPHLING_CONTAINERS=$(docker ps -aq -f name=morphling)

if [ -n "$REDIS_CONTAINERS" ]; then
    echo "Stopping and removing existing redis containers..."
    docker rm -f $REDIS_CONTAINERS
fi

if [ -n "$MORPHLING_CONTAINERS" ]; then
    echo "Stopping and removing existing morphling containers..."
    docker rm -f $MORPHLING_CONTAINERS
fi

# 2) (Optional) Kill any leftover run_devices.py processes
if pgrep -f "run_devices.py" >/dev/null; then
    echo "Killing leftover run_devices.py processes..."
    pkill -f "run_devices.py"
fi

# 3) Start a new Redis container
echo "Starting a new Redis container..."
docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

# 4) Generate device config
cd morphling/entrypoint
SPDLOG_LEVEL=debug python generate_device_config.py --num_devices 1 --device_type physical
cp device_config.json ../../scripts/

# 5) Run Morphling devices in the background
cd ../../scripts
SPDLOG_LEVEL=debug python run_devices.py \
    --num_devices 1 \
    --model_name facebook/opt-125m \
    --backend proxy \
    --seq_length 128 \
    --batch_size 1 \
    --cfg ../config/proxy/svr.ini \
    &

# 6) Start Nginx container (morphling-proxy) with the correct mounts for stream
cd ..
docker run -d \
    --name morphling-proxy \
    -p 443:443 \
    -v "$(pwd)/docker-nginx/nginx.conf:/etc/nginx/nginx.conf:ro" \
    -v "$(pwd)/docker-nginx/morphling_stream.conf:/etc/nginx/stream_conf.d/morphling_stream.conf:ro" \
    nginx:latest

echo "All done. Now test from local with: nc -vz <server_ip> 443"

# Keep script alive so the background job isn't killed
wait
```

## Paper evaluation pipeline (single-node RAN control + training)

This section is the human-reproducible path for generating the A/B evaluation
artifacts used by the paper subsection (Figure A, Figure B, and overhead table).

### Prerequisites

- Linux host with Docker and NVIDIA Container Toolkit (`--gpus all` required)
- GPU supported by this repo's runtime path (the current run used an NVIDIA L40S)
- Input traces present in `data/`:
  - `data/ldpc_trace_with_ctrl.csv`
  - `data/ldpc_trace_without_ctrl.csv`
- Network access for HuggingFace model/tokenizer download on first run

### 1) Validate trace inputs

```bash
python3 scripts/validate_traces.py --data-dir data
```

Expected result: both traces print `PASSED`.

### 2) Run the full pipeline

```bash
bash scripts/run_paper_experiments.sh
```

Expected completion marker:

```text
=== Done! Results in results ===
```

### 3) Verify generated artifacts

Core outputs:

- `results/paper_data.json`
- `results/table_overhead.json`
- `results/swap_timing_validation.json`
- `figures/fig_a_sm_timeline.pdf`
- `figures/fig_b_training_metrics.pdf`
- `figures/table_overhead.tex`

Per-run violation artifacts (for each run under both arms):

- `step_boundaries.csv`
- `gemm_log.csv`
- `violations.json`
- `violation_summary.csv`
- `violation_report.txt`

Quick check:

```bash
python3 - <<'PY'
import json
from pathlib import Path

paper = json.loads(Path("results/paper_data.json").read_text())
for arm in ("without_ctrl", "with_ctrl"):
    p = paper[arm]
    print(
        arm,
        "runs=", p.get("n_runs"),
        "tokens/s=", round(float(p.get("tokens_per_sec_mean", 0.0)), 3),
        "step_ms=", round(float(p.get("step_time_ms_mean", 0.0)), 3),
        "gflops=", round(float(p.get("gflops_mean", 0.0)), 3),
    )
PY
```

This pipeline now writes per-run violation-analysis artifacts directly from
`scripts/eval_greenctx_training.py` output:

- `step_boundaries.csv` (step-level wall-clock boundaries + `sm_count`)
- `gemm_log.csv` (autograd GEMM timing converted to ns)
- `violations.json` (machine-readable summary)
- `violation_summary.csv` and `violation_report.txt`

`scripts/aggregate_paper_results.py` consumes those artifacts to populate
overhead and violation fields in `results/paper_data.json`, including
`total_violations_mean`, `avg_violation_duration_us_mean`, and
`combined_overhead_ms`.

For the latest successful run in this worktree, both arms completed with
`n_runs = 5`, and all expected outputs above were generated.

Key CLI parameters for `scripts/eval_greenctx_training.py` are:

- `--trace-path` (not `--trace`)
- `--num-steps` (not `--steps`)
- `--skip-violation-analysis` to disable per-run violation outputs

## Troubleshooting

### Version mismatch or missing libraries
If you encounter errors like:

```bash
undefined symbol: sk_pop_free_ex (e.g., OpenSSL/gRPC mismatch)
ImportError: librttr_core.so.*: cannot open shared object file (missing RTTR)
ImportError: libmosquitto.so.*: cannot open shared object file (missing Mosquitto)
```

These typically mean libraries were built in a different environment than the
one used at runtime, or the dynamic linker cannot find them.

Fix steps:

```bash
# Rebuild within the current environment to match installed libraries
pip install --no-build-isolation --force-reinstall -e .
```

Set `LD_LIBRARY_PATH` if certain libraries (e.g., RTTR, Mosquitto) are in a
non-standard location:

```bash
# Example for RTTR
export LD_LIBRARY_PATH="/path/to/rttr/install/lib:$LD_LIBRARY_PATH"

# Example for local build artifacts (e.g., C++ .so files)
export LD_LIBRARY_PATH="/path/to/emulator/build/lib.linux-x86_64-cpython-310/morphling:$LD_LIBRARY_PATH"
```

## Per-GEMM green context switching

Morphling can optionally switch CUDA green contexts per linear GEMM using a
trace-driven controller and autograd hooks.

- Before each linear GEMM dispatch in forward and backward, the controller is
  asked to activate a partition with `activate_for_time(elapsed_us)`.
- `elapsed_us` is a wall-clock timestamp in microseconds taken from the first
  GEMM after the green context is set. Timestamps are relative to this epoch.
- After each GEMM completes, a structured log entry is recorded.

### Configuration

Configure a `GreenContextController` in wall-clock mode with a trace file:

```python
from morphling.runtime.green_context import (
    GreenContextConfig,
    GreenContextController,
)

cfg = GreenContextConfig(
    enabled=True,
    trace_path="traces/second_level.trace",
    clock_mode="wall",
)
controller = GreenContextController.from_config(device_id=0, cfg=cfg)
```

- `enabled=True` turns on green context management; otherwise an off backend
  is used and per-GEMM switching is disabled.
- `clock_mode="wall"` is required so `activate_for_time(elapsed_us)` uses
  wall-clock microseconds instead of step-based indices.
- `trace_path` must point to a trace that the selected backend can load.

### Hook usage and GEMM log format

Install linear hooks with the controller:

```python
from morphling.hooks import apply_hooks, get_gemm_log

apply_hooks("linear", greenctx=controller)
```

This routes linear GEMMs through a custom autograd function that:

- Activates the configured green context before each GEMM in
  forward, `backward_grad_input`, and `backward_grad_weight`.
- Deactivates the context after each GEMM.
- Appends a dictionary to an in-memory GEMM log for each operation.

Each log entry has the fields:

- `gemm_idx`: integer index of the GEMM within the current run.
- `phase`: one of `"forward"`, `"backward_grad_input"`, `"backward_grad_weight"`.
- `start_us`: wall-clock timestamp (microseconds) when `activate_for_time` was
  called.
- `end_us`: wall-clock timestamp (microseconds) after the GEMM completes.
- `duration_us`: `end_us - start_us`.
- `sm_count`: number of SMs requested for this GEMM, or `None` if the backend
  does not report a count.
- `greenctx_enabled`: boolean indicating whether a green context controller was
  configured when the entry was recorded.
- `m`: GEMM M dimension for this operation.
- `n`: GEMM N dimension for this operation.
- `k`: GEMM K dimension for this operation.

Example end-to-end usage:

```python
import torch

from morphling.runtime.green_context import (
    GreenContextConfig,
    GreenContextController,
)
from morphling.hooks import apply_hooks, get_gemm_log

cfg = GreenContextConfig(
    enabled=True,
    trace_path="traces/second_level.trace",
    clock_mode="wall",
)
controller = GreenContextController.from_config(device_id=0, cfg=cfg)

apply_hooks("linear", greenctx=controller)

model = torch.nn.Linear(16, 16).cuda()
x = torch.randn(8, 16, device="cuda", requires_grad=True)
target = torch.randn(8, 16, device="cuda")

out = model(x)
loss = torch.nn.functional.mse_loss(out, target)
loss.backward()

print(get_gemm_log()[0])
```

## C++ Tests

See `tests/cpp/README.md` for the full C++ test and benchmark catalog (unit,
CUDA/cuBLAS, XtGemm/worker, zerocopy, and benchmarks) plus run instructions.
The Docker image builds **all** C++ test categories, including the optional
XtGemm and zerocopy suites.

## Further Documentation

- `docs/DEV_README.md`
- `docs/DOCKER.md`
- `docs/GEMM_ID_ISSUES.md`
- `docs/EARLIEST_vs_LATEST.md`
- `tests/cpp/README.md`
- `tests/cpp/zerocopy/README.md`
