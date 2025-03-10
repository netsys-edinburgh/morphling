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
morphling_cmd save --model "facebook/opt-125m" --output <path to model checkpoint>
morphling_emulator --ckpt_path <path to model checkpoint>
SPDLOG_LEVEL=<level> MORPHLING_SERVER_ADDRESS=localhost:50051 MORPHLING_PIN_SIZE=10000000000  python tests/python/test_loaded_lib.py
```
