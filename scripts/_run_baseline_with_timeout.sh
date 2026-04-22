#!/usr/bin/env bash
set -euo pipefail

BASELINE="${1:?usage: _run_baseline_with_timeout.sh BASELINE [NDEV]}"
NDEV="${2:-4}"
OUTDIR="results/vtime/${BASELINE}"
CONTAINER_NAME="baseline_${BASELINE}_$$"

rm -rf "$OUTDIR" logs/perf_server* 2>/dev/null
mkdir -p "$OUTDIR/device_logs" logs

docker run --rm --gpus all \
  --name "$CONTAINER_NAME" \
  -v "$(pwd)/scripts:/app/scripts_host" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/config:/app/config" \
  -v "$(pwd)/logs:/app/logs" \
  -w /app \
  device-emulator:latest bash /app/scripts_host/_inner_baseline.sh "$BASELINE" "$NDEV" \
  > "/tmp/${BASELINE}_run.log" 2>&1 &

DOCKER_PID=$!

MAX_WAIT=600
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))

  if [ -f "$OUTDIR/vtime.log" ]; then
    EVENTS=$(grep -c "VTIME" "$OUTDIR/vtime.log" 2>/dev/null || echo 0)
    if [ "$EVENTS" -gt 100 ]; then
      echo "$BASELINE: VTIME log ready ($EVENTS events at ${ELAPSED}s), killing container"
      docker kill "$CONTAINER_NAME" 2>/dev/null || true
      wait $DOCKER_PID 2>/dev/null || true
      echo "$BASELINE: done"
      exit 0
    fi
  fi

  if ! kill -0 $DOCKER_PID 2>/dev/null; then
    echo "$BASELINE: container exited on its own at ${ELAPSED}s"
    break
  fi
done

if kill -0 $DOCKER_PID 2>/dev/null; then
  echo "$BASELINE: timed out at ${ELAPSED}s, killing"
  docker kill "$CONTAINER_NAME" 2>/dev/null || true
  wait $DOCKER_PID 2>/dev/null || true
fi

if [ -f "$OUTDIR/vtime.log" ]; then
  EVENTS=$(grep -c "VTIME" "$OUTDIR/vtime.log" 2>/dev/null || echo 0)
  echo "$BASELINE: $EVENTS VTIME events"
else
  echo "$BASELINE: MISSING vtime.log"
  tail -10 "/tmp/${BASELINE}_run.log" 2>/dev/null
fi
