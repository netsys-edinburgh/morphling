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

## Usage

```bash
if [ "$(docker ps -q -f name=redis)" ]; then
    docker stop redis
fi

docker run -dit --rm --name redis -p 6379:6379 redis
sleep 5

cd /scripts
SPDLOG_LEVEL=debug python run_devices.py     --num_devices 4     --model_name facebook/opt-125m     --backend proxy     --seq_length 128     --batch_size 1 --cfg ../config/proxy/svr.ini 
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

