# DeviceEmulator

## Installation

```bash
# use conda env or virtualenv
pip install -e .
```

## Usage

```bash
morphling_cmd save --model "facebook/opt-125m" --output <path to model checkpoint>
morphling_emulator --ckpt_path <path to model checkpoint>
SPDLOG_LEVEL=<level> MORPHLING_SERVER_ADDRESS=localhost:50051 MORPHLING_PIN_SIZE=10000000000  python tests/python/test_loaded_lib.py
```

# DeviceEmulator

## Installation

```bash
# Use conda env or virtualenv
pip install -e .

# If using conda, install libxslt (required by some torch dependencies on certain systems)
conda install -c conda-forge libxslt

# If you have CUDA available and want to ensure Torch can detect it:
export CUDA_HOME=/usr/local/cuda

# If you run into Torch import errors or build issues, try building without isolation:
pip install --no-build-isolation .
```

## Virtual Device Usage

```bash
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

cd /scripts
SPDLOG_LEVEL=debug python run_devices.py     --num_devices 4     --model_name facebook/opt-125m     --backend proxy     --seq_length 128     --batch_size 1 --cfg ../config/proxy/svr.ini
```
## Physical Device Usage

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
cd /home/eren/Emulator/DeviceEmulator/morphling/entrypoint
SPDLOG_LEVEL=debug python generate_device_config.py --num_devices 1 --device_type physical
cp device_config.json /home/eren/Emulator/DeviceEmulator/scripts/

# 5) Run Morphling devices in the background
echo "Starting devices..."
cd /home/eren/Emulator/DeviceEmulator/scripts
SPDLOG_LEVEL=debug python run_devices.py \
    --num_devices 1 \
    --model_name facebook/opt-125m \
    --backend proxy \
    --seq_length 128 \
    --batch_size 1 \
    --cfg /home/eren/Emulator/DeviceEmulator/config/proxy/svr.ini \
    &  # <-- run in background

# 6) Start Nginx container (morphling-proxy) with the correct mounts for stream
echo "Starting Nginx container (morphling-proxy)..."
cd /home/eren/Emulator/DeviceEmulator
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
## Trouble Shooting

Version Mismatch or Missing Libraries
If you encounter errors like:

```bash
undefined symbol: sk_pop_free_ex (e.g., OpenSSL/gRPC mismatch)
ImportError: librttr_core.so.*: cannot open shared object file (missing RTTR)
ImportError: libmosquitto.so.*: cannot open shared object file (missing Mosquitto)
```
these typically mean libraries were built in a different environment than the one used at runtime, or the dynamic linker can’t find them.

Steps to Fix:

Reinstall / Rebuild Without Isolation
Rebuild within the current environment to match installed libraries:

```bash
pip install --no-build-isolation --force-reinstall -e .
```
Set LD_LIBRARY_PATH if certain libraries (e.g., RTTR, Mosquitto) are in a non-standard location:
```bash
# Example for RTTR
export LD_LIBRARY_PATH="/path/to/rttr/install/lib:$LD_LIBRARY_PATH"

# Example for local build artifacts (e.g., C++ .so files)
export LD_LIBRARY_PATH="/path/to/emulator/build/lib.linux-x86_64-cpython-310/morphling:$LD_LIBRARY_PATH"
```
