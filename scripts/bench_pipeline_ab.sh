#!/usr/bin/env bash
set -euo pipefail

MODE_A="${1:-legacy}"
MODE_B="${2:-dual}"
REPETITIONS="${REPETITIONS:-5}"
OUT_DIR="results/pipeline_ab/$(date +%Y%m%d_%H%M%S)"
BENCH_BIN="tests/cpp/build/bench_dual_stream_gemm"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "ERROR: benchmark binary not found or not executable: $BENCH_BIN"
  echo "Hint: build with ENABLE_PIPELINE_TESTS=ON"
  exit 1
fi

run_mode() {
  local mode="$1"
  local out_json="$OUT_DIR/${mode}.json"

  echo "--- Warmup ${mode} (${WARMUP_SECONDS}s) ---"
  MORPHLING_WORKER_PIPELINE="$mode" timeout "${WARMUP_SECONDS}s" "$BENCH_BIN" \
    --benchmark_filter='BM_Gemm_SingleTask/1024' \
    --benchmark_min_time=0.1 \
    --benchmark_format=json \
    --benchmark_out="$OUT_DIR/${mode}_warmup.json" || true

  echo "--- Running mode ${mode} ---"
  MORPHLING_WORKER_PIPELINE="$mode" "$BENCH_BIN" \
    --benchmark_format=json \
    --benchmark_repetitions="$REPETITIONS" \
    --benchmark_out="$out_json"
}

echo "=== A/B Test: ${MODE_A} vs ${MODE_B} ==="
echo "Repetitions: ${REPETITIONS}"
echo "Output: ${OUT_DIR}"

run_mode "$MODE_A"
run_mode "$MODE_B"

python3 scripts/compare_bench_json.py \
  "$OUT_DIR/${MODE_A}.json" \
  "$OUT_DIR/${MODE_B}.json" \
  --output "$OUT_DIR/comparison.json"

echo "=== Comparison written to $OUT_DIR/comparison.json ==="
