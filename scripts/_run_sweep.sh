#!/usr/bin/env bash
# Run VTIME emulation for all (model, baseline) pairs.
set -euo pipefail

MODELS=("opt-1.3b" "llama2-7b" "opt-13b")
BASELINES=("cleave" "dtfm" "asteroid" "confident" "alpa")
NDEV=4
FLEET="results/comparison/generated_device_fleet.json"

mkdir -p results/vtime_sweep

for model in "${MODELS[@]}"; do
  MANIFEST_DIR="results/comparison_${model}/manifests"
  if [ ! -d "$MANIFEST_DIR" ]; then
    echo "[$model] MISSING manifest dir $MANIFEST_DIR, skipping"
    continue
  fi

  for bl in "${BASELINES[@]}"; do
    MANIFEST="$MANIFEST_DIR/${bl}_manifest.json"
    OUTDIR="results/vtime_sweep/${model}/${bl}"

    if [ ! -f "$MANIFEST" ]; then
      echo "[$model/$bl] MISSING manifest, skipping"
      continue
    fi

    rm -rf "$OUTDIR" logs/perf_server* 2>/dev/null || true
    mkdir -p "$OUTDIR/device_logs" logs

    echo "[$model/$bl] starting emulator..."
    docker run --rm --gpus all \
      --name "sweep_${model//[.\/]/_}_${bl}" \
      -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      -v "$(pwd)/scripts:/app/scripts_host" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/config:/app/config" \
      -v "$(pwd)/logs:/app/logs" \
      -w /app \
      device-emulator:latest bash /app/scripts_host/_inner_baseline.sh \
        "$bl" "$NDEV" "$FLEET" "$MANIFEST" "$OUTDIR" \
      > "/tmp/sweep_${model//[.\/]/_}_${bl}.log" 2>&1 &
    DOCKER_PID=$!

    for i in $(seq 1 120); do
      sleep 5
      if [ -f "$OUTDIR/vtime.log" ]; then
        EVENTS=$(grep -c "VTIME" "$OUTDIR/vtime.log" 2>/dev/null || echo 0)
        if [ "$EVENTS" -gt 50 ]; then
          docker kill "sweep_${model//[.\/]/_}_${bl}" 2>/dev/null || true
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
      tail -5 "/tmp/sweep_${model//[.\/]/_}_${bl}.log"
    fi
  done
done

echo "=== Sweep complete ==="
