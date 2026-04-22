#!/usr/bin/env bash
set -euo pipefail

BASELINE=${1:?missing baseline}
MANIFEST=${2:?missing manifest json}
FLEET=${3:?missing fleet json}
OUTPUT_DIR=${4:?missing output dir}
IMAGE="device-emulator:latest"

mkdir -p "$OUTPUT_DIR"

NUM_DEVICES=$(python3 -c "import json; print(len(json.load(open('$FLEET', 'r', encoding='utf-8'))))")

DOCKER_TIMEOUT=$(( ${VTIME_TIMEOUT:-900} + 120 ))

mkdir -p logs
timeout --signal=KILL "${DOCKER_TIMEOUT}s" \
docker run --rm --init --gpus all \
  -e MORPHLING_NO_GREEN_CTX="${MORPHLING_NO_GREEN_CTX:-1}" \
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

    python3 /app/scripts_host/replay_manifest.py \
      --manifest '$MANIFEST' \
      --cfg config/proxy/svr.ini \
      --num-devices '$NUM_DEVICES' \
      --output-log '$OUTPUT_DIR/vtime.log' \
      --timeout ${VTIME_TIMEOUT:-900} &
    SVR_PID=\$!
    sleep ${VTIME_STARTUP_WAIT:-3}

    NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l)
    NUM_GPUS=\${NUM_GPUS:-1}

    DEVICE_PARAMS=\$(python3 -c \"
import json
fleet = json.load(open('$FLEET'))
n = len(fleet)
for i in range($NUM_DEVICES):
    d = fleet[i % n]
    print(int(d['flops']), int(d['memory']),
          int(d['ul_bw']), int(d['dl_bw']),
          d.get('ul_lat', 0.0), d.get('dl_lat', 0.0))
\")

    DEV_IDX=0
    while IFS=' ' read -r FLOPS MEMORY UL_BW DL_BW UL_LAT DL_LAT; do
      GPU_IDX=\$(( DEV_IDX % NUM_GPUS ))
      CUDA_VISIBLE_DEVICES=\$GPU_IDX \
      morphling_device --id \$DEV_IDX \
        --flops \"\$FLOPS\" --memory \"\$MEMORY\" \
        --ul_bw \"\$UL_BW\" --dl_bw \"\$DL_BW\" \
        --ul_lat \"\$UL_LAT\" --dl_lat \"\$DL_LAT\" \
        --backend proxy --cfg config/proxy/svr.ini \
        > \"$OUTPUT_DIR/device_logs/dev_\${DEV_IDX}.log\" 2>&1 &
      DEV_IDX=\$((DEV_IDX + 1))
    done <<< \"\$DEVICE_PARAMS\"

    wait \$SVR_PID
    RET=\$?

    kill \$(jobs -p) 2>/dev/null || true
    sleep 2
    kill -9 \$(jobs -p) 2>/dev/null || true
    disown -a 2>/dev/null || true
    exit \$RET
  "

echo "[$BASELINE] Done: $OUTPUT_DIR/vtime.log"
