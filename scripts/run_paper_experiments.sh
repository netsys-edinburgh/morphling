#!/usr/bin/env bash
set -euo pipefail

RUNS=5
SEED_BASE=42
TOTAL_SMS=48
MAX_TRACE_SLOTS=27462
RESULTS_DIR="results"
IMAGE="device-emulator:latest"

echo "=== Paper Experiment Runner ==="
echo "Runs per config: $RUNS"
echo "Seed base: $SEED_BASE"

echo -e "\n--- Step 1: Building Docker image ---"
docker build -t "$IMAGE" .

echo -e "\n--- Step 2: Extracting GEMM shapes ---"
docker run --rm --gpus all \
    -v "$(pwd)/results:/app/results" \
    "$IMAGE" python3 scripts/eval_greenctx_training.py \
    --steps 3 --trace data/ldpc_trace_with_ctrl.csv \
    --dump-gemm-shapes --output-dir results

echo -e "\n--- Step 3: GFLOPS benchmark ---"
docker run --rm --gpus all \
    -v "$(pwd)/results:/app/results" \
    "$IMAGE" python3 scripts/bench_gflops_per_sm.py \
    --gemm-shapes results/gemm_shapes.json \
    --output results/gflops_per_sm.json

for config in "without_ctrl" "with_ctrl"; do
    trace="data/ldpc_trace_${config}.csv"
    echo -e "\n--- Config: $config (trace: $trace) ---"

    for i in $(seq 0 $((RUNS - 1))); do
        seed=$((SEED_BASE + i))
        outdir="$RESULTS_DIR/${config}/run_${i}"
        echo "  Run $i (seed=$seed) -> $outdir"

        docker run --rm --gpus all \
            -v "$(pwd)/results:/app/results" \
            -v "$(pwd)/data:/app/data" \
            "$IMAGE" python3 scripts/eval_greenctx_training.py \
            --runs 1 --seed-base "$seed" \
            --max-trace-slots "$MAX_TRACE_SLOTS" \
            --trace "$trace" \
            --total-sms "$TOTAL_SMS" \
            --output-dir "$outdir" || {
                echo "  WARNING: Run $i failed, continuing..."
                continue
            }
    done
done

echo -e "\n--- Step 5: Cross-validate swap timing ---"
docker run --rm --gpus all \
    -v "$(pwd)/results:/app/results" \
    -v "$(pwd)/data:/app/data" \
    "$IMAGE" python3 scripts/cross_validate_swap_timing.py \
    --trace data/ldpc_trace_with_ctrl.csv \
    --output results/swap_timing_validation.json \
    --steps 5

echo -e "\n--- Step 6: Aggregating results ---"
python3 scripts/aggregate_paper_results.py \
    --results-dir "$RESULTS_DIR" \
    --gflops results/gflops_per_sm.json \
    --output results/paper_data.json

echo -e "\n=== Done! Results in $RESULTS_DIR ==="
