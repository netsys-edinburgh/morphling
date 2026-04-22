#!/usr/bin/env bash
set -euo pipefail

BASELINE="$1"
NDEV="$2"
FLEET="${3:-results/comparison/generated_device_fleet.json}"
MANIFEST="${4:-results/comparison/manifests/${BASELINE}_manifest.json}"
OUTDIR="${5:-results/vtime/${BASELINE}}"

mkdir -p "$OUTDIR/device_logs" logs

python3 /app/scripts_host/replay_manifest.py \
  --manifest "$MANIFEST" \
  --cfg config/proxy/svr.ini \
  --num-devices "$NDEV" \
  --output-log "$OUTDIR/vtime.log" \
  --timeout 180 &
SVR_PID=$!
sleep "${INNER_STARTUP_WAIT:-2}"

# Read all device params in a single Python call (avoids 6N interpreter boots)
DEVICE_PARAMS=$(python3 -c "
import json, sys
fleet = json.load(open('$FLEET'))
n = len(fleet)
for i in range($NDEV):
    d = fleet[i % n]
    print(int(d['flops']), int(d['memory']),
          int(d['ul_bw']), int(d['dl_bw']),
          d.get('ul_lat', 0.0), d.get('dl_lat', 0.0))
")

DEV_IDX=0
while IFS=' ' read -r FLOPS MEMORY UL_BW DL_BW UL_LAT DL_LAT; do
  morphling_device --id $DEV_IDX \
    --flops "$FLOPS" --memory "$MEMORY" \
    --ul_bw "$UL_BW" --dl_bw "$DL_BW" \
    --ul_lat "$UL_LAT" --dl_lat "$DL_LAT" \
    --backend proxy --cfg config/proxy/svr.ini \
    > "$OUTDIR/device_logs/dev_${DEV_IDX}.log" 2>&1 &
  DEV_IDX=$((DEV_IDX + 1))
done <<< "$DEVICE_PARAMS"

wait $SVR_PID
RET=$?
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
exit $RET
