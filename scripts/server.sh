#!/bin/bash

# Server launch script
python3 scripts/run_server.py \
  --backend proxy \
  --model_name facebook/opt-125m \
  --enable-hooks