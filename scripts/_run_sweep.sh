#!/usr/bin/env bash
# Run VTIME emulation for all (model, baseline) pairs.
set -euo pipefail

MODELS=("opt-1.3b" "llama2-7b" "opt-13b")
BASELINES=("cleave" "dtfm" "asteroid" "confident" "alpa")
NDEV=4
FLEET="results/comparison/generated_device_fleet.json"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

mkdir -p results/vtime_sweep

run_one_baseline() {
  local model="$1" bl="$2"
  local MANIFEST_DIR="results/comparison_${model}/manifests"
  local MANIFEST="$MANIFEST_DIR/${bl}_manifest.json"
  local OUTDIR="results/vtime_sweep/${model}/${bl}"
  local CONTAINER="sweep_${model//[.\/]/_}_${bl}"
  local LOGFILE="/tmp/sweep_${model//[.\/]/_}_${bl}.log"

  if [ ! -f "$MANIFEST" ]; then
    echo "[$model/$bl] MISSING manifest, skipping"
    return 0
  fi

  rm -rf "$OUTDIR" 2>/dev/null || true
  mkdir -p "$OUTDIR/device_logs"

  echo "[$model/$bl] starting emulator..."
  docker run --rm --gpus all \
    --name "$CONTAINER" \
    -v "$(pwd)/scripts:/app/scripts_host" \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/logs:/app/logs" \
    -w /app \
    device-emulator:latest bash /app/scripts_host/_inner_baseline.sh \
      "$bl" "$NDEV" "$FLEET" "$MANIFEST" "$OUTDIR" \
    > "$LOGFILE" 2>&1 &
  local DOCKER_PID=$!

  for _ in $(seq 1 120); do
    sleep 5
    if [ -f "$OUTDIR/vtime.log" ]; then
      local EVENTS
      EVENTS=$(grep -c "VTIME" "$OUTDIR/vtime.log" 2>/dev/null || echo 0)
      if [ "$EVENTS" -gt 50 ]; then
        docker kill "$CONTAINER" 2>/dev/null || true
        wait $DOCKER_PID 2>/dev/null || true
        break
      fi
    fi
    if ! kill -0 $DOCKER_PID 2>/dev/null; then
      break
    fi
  done

  if [ -f "$OUTDIR/vtime.log" ]; then
    echo "[$model/$bl] done: $(grep -c VTIME "$OUTDIR/vtime.log") VTIME events"
  else
    echo "[$model/$bl] FAILED"
    tail -5 "$LOGFILE"
  fi
}

for model in "${MODELS[@]}"; do
  MANIFEST_DIR="results/comparison_${model}/manifests"
  if [ ! -d "$MANIFEST_DIR" ]; then
    echo "[$model] MISSING manifest dir $MANIFEST_DIR, skipping"
    continue
  fi

  rm -f logs/perf_server* 2>/dev/null || true
  mkdir -p logs

  RUNNING=0
  for bl in "${BASELINES[@]}"; do
    run_one_baseline "$model" "$bl" &
    RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -ge "$MAX_PARALLEL" ]; then
      wait -n 2>/dev/null || true
      RUNNING=$((RUNNING - 1))
    fi
  done
  wait
done

echo "=== Sweep complete ==="
