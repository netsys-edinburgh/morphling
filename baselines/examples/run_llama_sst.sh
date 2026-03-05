#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERCEPT_DIR="${SCRIPT_DIR}/gemm_intercept"
INTERCEPT_SO="${INTERCEPT_DIR}/libgemm_intercept.so"
TRACE_FILE="${SCRIPT_DIR}/sample_greenctx_trace.csv"
TRAIN_SCRIPT="${SCRIPT_DIR}/llama_sst_single_gpu.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_violations.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/llama_sst_single}"
MAX_SM="${MAX_SM_COUNT:-48}"
MODE="${1:-both}"
shift || true

if [[ ! -f "${INTERCEPT_SO}" ]]; then
  make -C "${INTERCEPT_DIR}"
fi

case "${MODE}" in
  basic)
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --max-sm-count "${MAX_SM}" \
      "$@"
    ;;
  greenctx)
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --greenctx-enabled \
      --greenctx-trace "${TRACE_FILE}" \
      --max-sm-count "${MAX_SM}" \
      "$@"
    ;;
  gemm)
    GEMM_LOG_PATH="${GEMM_LOG_PATH:-${OUTPUT_DIR}/gemm_log.csv}" \
    LD_PRELOAD="${INTERCEPT_SO}:/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --gemm-log "${GEMM_LOG_PATH:-${OUTPUT_DIR}/gemm_log.csv}" \
      --max-sm-count "${MAX_SM}" \
      "$@"
    ;;
  both)
    GEMM_LOG_PATH="${GEMM_LOG_PATH:-${OUTPUT_DIR}/gemm_log.csv}" \
    LD_PRELOAD="${INTERCEPT_SO}:/usr/local/cuda/lib64/libcublas.so.12:/usr/local/cuda/lib64/libcublasLt.so.12${LD_PRELOAD:+:${LD_PRELOAD}}" \
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --greenctx-enabled \
      --greenctx-trace "${TRACE_FILE}" \
      --gemm-log "${GEMM_LOG_PATH:-${OUTPUT_DIR}/gemm_log.csv}" \
      --max-sm-count "${MAX_SM}" \
      "$@"
    ;;
  analyze)
    GEMM_LOG="${GEMM_LOG_PATH:-${OUTPUT_DIR}/gemm_log.csv}"
    STEP_LOG="${OUTPUT_DIR}/step_boundaries.csv"
    if [[ ! -f "${GEMM_LOG}" ]]; then
      echo "ERROR: GEMM log not found: ${GEMM_LOG}" >&2
      echo "Run with 'both' mode first." >&2
      exit 1
    fi
    if [[ ! -f "${STEP_LOG}" ]]; then
      echo "ERROR: Step boundaries not found: ${STEP_LOG}" >&2
      echo "Run with 'both' or 'greenctx' mode first." >&2
      exit 1
    fi
    "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" \
      --gemm-log "${GEMM_LOG}" \
      --step-log "${STEP_LOG}" \
      --trace "${TRACE_FILE}" \
      --max-sm-count "${MAX_SM}" \
      --output "${OUTPUT_DIR}/violation_report.txt" \
      --output-csv "${OUTPUT_DIR}/violation_summary.csv" \
      "$@"
    echo "Report: ${OUTPUT_DIR}/violation_report.txt"
    echo "CSV:    ${OUTPUT_DIR}/violation_summary.csv"
    ;;
  *)
    echo "Usage: $0 [basic|greenctx|gemm|both|analyze] [extra-args...]" >&2
    echo "" >&2
    echo "Modes:" >&2
    echo "  basic    - Train without green context or GEMM interception" >&2
    echo "  greenctx - Train with green context SM partitioning" >&2
    echo "  gemm     - Train with GEMM timing interception" >&2
    echo "  both     - Train with green context + GEMM interception" >&2
    echo "  analyze  - Run violation analysis on existing logs" >&2
    echo "" >&2
    echo "Environment:" >&2
    echo "  OUTPUT_DIR     output directory (default: ./output/llama_sst_single)" >&2
    echo "  MAX_SM_COUNT   max SM count threshold (default: 48)" >&2
    exit 2
    ;;
esac
