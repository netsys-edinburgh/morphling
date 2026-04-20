#!/usr/bin/env bash
# Run VTIME emulation for multiple models x baselines, then post-process.
#
# Usage:
#   bash scripts/run_all_vtime_models.sh
#
# Environment overrides:
#   VTIME_TIMEOUT=900       replay_manifest timeout (seconds)
#   VTIME_STARTUP_WAIT=10   sleep before launching devices
#   MODELS="opt-1.3b opt-13b"  subset of models to run
#   BASELINES="cleave dtfm"    subset of baselines to run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export VTIME_TIMEOUT="${VTIME_TIMEOUT:-1800}"
export VTIME_STARTUP_WAIT="${VTIME_STARTUP_WAIT:-10}"

# ── Model → manifest directory mapping ──────────────────────────
declare -A MODEL_DIRS=(
  [opt-125m]="results/comparison_opt-125m"
  [opt-1.3b]="results/comparison_opt-1.3b"
  [opt-13b]="results/comparison_opt-13b"
  [opt-30b]="results/comparison_opt-30b"
  [opt-66b]="results/comparison_opt-66b"
  [llama2-7b]="results/comparison_llama2-7b"
  [llama2-13b]="results/comparison_llama2-13b"
  [llama2-70b]="results/comparison_llama2-70b"
)

# Default fleet (64 devices).  All comparison_* runs used 64.
FLEET="results/comparison/generated_device_fleet.json"

MODELS="${MODELS:-opt-125m opt-1.3b opt-13b opt-30b opt-66b llama2-7b llama2-13b llama2-70b}"
BASELINES="${BASELINES:-cleave dtfm asteroid confident alpa}"

VTIME_ROOT="results/vtime_models"

# ── Phase 1: Emulation ──────────────────────────────────────────
echo "=== Phase 1: VTIME Emulation ==="
echo "    Models   : $MODELS"
echo "    Baselines: $BASELINES"
echo "    Timeout  : ${VTIME_TIMEOUT}s"
echo ""

for model in $MODELS; do
  dir="${MODEL_DIRS[$model]:-}"
  if [ -z "$dir" ]; then
    echo "[ERROR] Unknown model: $model"
    exit 1
  fi

  for baseline in $BASELINES; do
    manifest="$dir/manifests/${baseline}_manifest.json"
    outdir="$VTIME_ROOT/$model/$baseline"

    if [ ! -f "$manifest" ]; then
      echo "[SKIP] Missing manifest: $manifest"
      continue
    fi

    # Skip if vtime.log already exists (resume support)
    if [ -f "$outdir/vtime.log" ]; then
      echo "[SKIP] Already done: $outdir/vtime.log"
      continue
    fi

    echo "[RUN]  $model / $baseline  (timeout=${VTIME_TIMEOUT}s)"
    ts_start=$(date +%s)

    if bash scripts/run_vtime_experiment.sh \
        "$baseline" "$manifest" "$FLEET" "$outdir"; then
      elapsed=$(( $(date +%s) - ts_start ))
      echo "[OK]   $model / $baseline  (${elapsed}s)"
    else
      elapsed=$(( $(date +%s) - ts_start ))
      echo "[FAIL] $model / $baseline  (${elapsed}s)"
    fi
  done
done

# ── Phase 2: Post-process ───────────────────────────────────────
echo ""
echo "=== Phase 2: Post-Processing ==="

for model in $MODELS; do
  model_outdir="$VTIME_ROOT/$model"

  for baseline in $BASELINES; do
    vtlog="$model_outdir/$baseline/vtime.log"
    if [ ! -f "$vtlog" ]; then
      echo "[SKIP] No vtime log: $vtlog"
      continue
    fi

    echo "[POST] $model / $baseline"
    python3 scripts/run_baseline_comparison.py \
      --model "$model" \
      --num-devices 64 \
      --batch-size 16 \
      --seq-len 512 \
      --micro-batch-size 1 \
      --baselines "$baseline" \
      --vtime-log "$vtlog" \
      --output-dir "$model_outdir/${baseline}_results" \
      --log-level WARNING \
    || echo "[WARN] Post-processing failed for $model / $baseline"
  done
done

# ── Phase 3: Aggregate ──────────────────────────────────────────
echo ""
echo "=== Phase 3: Aggregate ==="
python3 scripts/aggregate_vtime_results.py \
  --vtime-root "$VTIME_ROOT" \
  --output "$VTIME_ROOT/vtime_summary.json"

echo ""
echo "Done.  Summary at: $VTIME_ROOT/vtime_summary.json"
