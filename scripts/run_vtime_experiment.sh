#!/usr/bin/env bash
set -euo pipefail

BASELINE=${1:?missing baseline}
MANIFEST=${2:?missing manifest json}
FLEET=${3:?missing fleet json}
OUTPUT_DIR=${4:?missing output dir}
IMAGE="device-emulator:latest"

mkdir -p "$OUTPUT_DIR"

NUM_DEVICES=$(python3 -c "import json; print(len(json.load(open('$FLEET', 'r', encoding='utf-8'))))")

mkdir -p logs
docker run --rm --gpus all \
  -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  -v "$(pwd)/scripts:/app/scripts_host" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/logs:/app/logs" \
  -w /app \
  "$IMAGE" bash -lc "
    set -euo pipefail
    mkdir -p '$OUTPUT_DIR'
    mkdir -p '$OUTPUT_DIR/device_logs'
    mkdir -p logs

    # 1. Start server (replay_manifest) in background — it creates a dynamic
    #    svr config with correct num_device/barrier_count, then waits for devices
    python3 /app/scripts_host/replay_manifest.py \
      --manifest '$MANIFEST' \
      --cfg config/proxy/svr.ini \
      --num-devices '$NUM_DEVICES' \
      --output-log '$OUTPUT_DIR/vtime.log' \
      --timeout 120 &
    SVR_PID=\$!
    sleep 5

    # 2. Start devices — server is already listening. Pass server config so
    #    the C++ ProxyCli reads the correct listen_ip/port from it.
    for i in \$(seq 0 \$(($NUM_DEVICES - 1))); do
      morphling_device --id \$i --flops 5T --memory 2G \
        --ul_bw 5M --dl_bw 50M --ul_lat 0.0 --dl_lat 0.0 \
        --backend proxy --cfg config/proxy/svr.ini \
        > '$OUTPUT_DIR/device_logs/dev_\${i}.log' 2>&1 &
    done

    # 3. Wait for server to finish dispatching
    wait \$SVR_PID
    RET=\$?

    # 4. Cleanup devices
    kill \$(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    exit \$RET
  "

echo "[$BASELINE] Done: $OUTPUT_DIR/vtime.log"
