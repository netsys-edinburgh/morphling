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