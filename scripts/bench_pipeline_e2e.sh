#!/usr/bin/env bash
set -euo pipefail

MODE_A="${1:-legacy}"
MODE_B="${2:-dual}"
NUM_DEVICES="${NUM_DEVICES:-4}"
NUM_MATMULS="${NUM_MATMULS:-16}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CFG_PATH="${CFG_PATH:-config/proxy/svr.ini}"
OUT_DIR="results/pipeline_e2e/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

run_mode() {
  local mode="$1"
  local out_json="$OUT_DIR/${mode}_e2e.json"

  echo "--- Running E2E mode ${mode} ---"
  MORPHLING_WORKER_PIPELINE="$mode" \
    python3 scripts/run_devices.py \
      --num_devices "$NUM_DEVICES" \
      --backend proxy \
      --seq_length "$SEQ_LEN" \
      --batch_size "$BATCH_SIZE" \
      --cfg "$CFG_PATH" \
      --benchmark_mode \
      --concurrent_submit \
      --num_matmuls "$NUM_MATMULS" \
      --output_json "$out_json"
}

echo "=== E2E A/B Test: ${MODE_A} vs ${MODE_B} ==="
echo "Devices=${NUM_DEVICES}, Matmuls/device=${NUM_MATMULS}, SeqLen=${SEQ_LEN}, Batch=${BATCH_SIZE}"
echo "Output: ${OUT_DIR}"

run_mode "$MODE_A"
run_mode "$MODE_B"

echo "=== E2E results in ${OUT_DIR} ==="
