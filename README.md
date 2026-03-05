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

- `DEV_README.md`
- `DOCKER.md`
- `GEMM_ID_ISSUES.md`
- `EARLIEST_vs_LATEST.md`
- `tests/cpp/README.md`
- `tests/cpp/zerocopy/README.md`
