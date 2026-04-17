#!/usr/bin/env bash
set -euo pipefail

BASELINE=${1:?usage: run_single_baseline.sh BASELINE}
NDEV=${2:-4}
MANIFEST="results/comparison/manifests/${BASELINE}_manifest.json"
OUTDIR="results/vtime/${BASELINE}"

rm -rf "$OUTDIR" logs/perf_server* 2>/dev/null
mkdir -p "$OUTDIR/device_logs" logs

docker run --rm --gpus all \
  -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  -v "$(pwd)/scripts:/app/scripts_host" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/logs:/app/logs" \
  -w /app \
  device-emulator:latest bash /app/scripts_host/_inner_baseline.sh "$BASELINE" "$NDEV"

if [ -f "$OUTDIR/vtime.log" ]; then
  EVENTS=$(grep -c VTIME "$OUTDIR/vtime.log" 2>/dev/null || echo 0)
  echo "$BASELINE: $EVENTS VTIME events"
else
  echo "$BASELINE: MISSING"
fi
