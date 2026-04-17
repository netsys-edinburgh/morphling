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
sleep 5

for i in $(seq 0 $(($NDEV - 1))); do
  FLOPS=$(python3 -c "import json; f=json.load(open('$FLEET')); print(int(f[$i % len(f)]['flops']))")
  MEMORY=$(python3 -c "import json; f=json.load(open('$FLEET')); print(int(f[$i % len(f)]['memory']))")
  UL_BW=$(python3 -c "import json; f=json.load(open('$FLEET')); print(int(f[$i % len(f)]['ul_bw']))")
  DL_BW=$(python3 -c "import json; f=json.load(open('$FLEET')); print(int(f[$i % len(f)]['dl_bw']))")
  UL_LAT=$(python3 -c "import json; f=json.load(open('$FLEET')); print(f[$i % len(f)].get('ul_lat', 0.0))")
  DL_LAT=$(python3 -c "import json; f=json.load(open('$FLEET')); print(f[$i % len(f)].get('dl_lat', 0.0))")

  morphling_device --id $i \
    --flops "$FLOPS" --memory "$MEMORY" \
    --ul_bw "$UL_BW" --dl_bw "$DL_BW" \
    --ul_lat "$UL_LAT" --dl_lat "$DL_LAT" \
    --backend proxy --cfg config/proxy/svr.ini \
    > "$OUTDIR/device_logs/dev_${i}.log" 2>&1 &
done

wait $SVR_PID
RET=$?
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
exit $RET
