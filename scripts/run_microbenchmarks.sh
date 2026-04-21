#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${REPO_ROOT}/results/microbenchmarks"
FIG_DIR="${REPO_ROOT}/figures/evaluation"

mkdir -p "${RESULTS_DIR}" "${FIG_DIR}"

echo "[1/6] Running placement microbenchmark"
python3 "${SCRIPT_DIR}/microbench_placement.py" \
  --output-json "${RESULTS_DIR}/placement_overhead.json"

echo "[2/6] Running dispatch microbenchmark"
python3 "${SCRIPT_DIR}/microbench_dispatch.py" \
  --output-json "${RESULTS_DIR}/dispatch_overhead.json"

echo "[3/6] Running recovery microbenchmark"
python3 "${SCRIPT_DIR}/microbench_recovery.py" \
  --output-json "${RESULTS_DIR}/recovery_breakdown.json"

echo "[4/6] Generating placement table"
python3 "${SCRIPT_DIR}/plot_micro_placement.py" \
  --input-json "${RESULTS_DIR}/placement_overhead.json" \
  --output-tex "${FIG_DIR}/table_micro_placement.tex"

echo "[5/6] Generating dispatch figure"
python3 "${SCRIPT_DIR}/plot_micro_dispatch.py" \
  --input-json "${RESULTS_DIR}/dispatch_overhead.json" \
  --output-pdf "${FIG_DIR}/micro_dispatch.pdf"

echo "[6/6] Generating recovery figure"
python3 "${SCRIPT_DIR}/plot_micro_recovery.py" \
  --input-json "${RESULTS_DIR}/recovery_breakdown.json" \
  --output-pdf "${FIG_DIR}/micro_recovery.pdf"

echo
echo "Microbenchmarks complete."
echo "Results:"
echo "  - ${RESULTS_DIR}/placement_overhead.json"
echo "  - ${RESULTS_DIR}/dispatch_overhead.json"
echo "  - ${RESULTS_DIR}/recovery_breakdown.json"
echo "Figures/Tables:"
echo "  - ${FIG_DIR}/table_micro_placement.tex"
echo "  - ${FIG_DIR}/micro_dispatch.pdf"
echo "  - ${FIG_DIR}/micro_dispatch.png"
echo "  - ${FIG_DIR}/micro_recovery.pdf"
echo "  - ${FIG_DIR}/micro_recovery.png"
