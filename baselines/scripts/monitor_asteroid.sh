#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NAMESPACE="${NAMESPACE:-default}"
LABEL="app=baselines-asteroid"
REFRESH_SECONDS="${REFRESH_SECONDS:-2}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() {
  echo -e "${BLUE}[monitor-asteroid]${NC} $*"
}

warn() {
  echo -e "${YELLOW}[monitor-asteroid]${NC} $*"
}

err() {
  echo -e "${RED}[monitor-asteroid]${NC} $*" >&2
}

banner() {
  echo -e "${BLUE}====================================================${NC}"
  echo -e "${BLUE}BASELINES-ASTEROID MONITOR${NC}"
  echo -e "${BLUE}====================================================${NC}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [mode] [options]

Modes:
  dashboard        Auto-refresh dashboard (default)
  once             Print one-time snapshot
  logs [target]    Stream logs for pod or rank
  rank <id>        Show details for one rank
  tensorboard      Sync logs and launch TensorBoard via deploy script

Targets:
  [target] for logs can be a pod name or numeric rank

Options:
  -n, --namespace <ns>   Kubernetes namespace (default: ${NAMESPACE})
  -l, --label <selector> Label selector (default: ${LABEL})
  -r, --refresh <sec>    Dashboard refresh period (default: ${REFRESH_SECONDS})
  -h, --help             Show this help message

Examples:
  $(basename "$0")
  $(basename "$0") dashboard --namespace default
  $(basename "$0") once
  $(basename "$0") logs 0
  $(basename "$0") rank 2
  $(basename "$0") tensorboard
EOF
}

get_pods() {
  kubectl get pods \
    -n "${NAMESPACE}" \
    -l "${LABEL}" \
    --sort-by=.metadata.name \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
}

get_pod_for_rank() {
  local rank="$1"
  kubectl get pods \
    -n "${NAMESPACE}" \
    -l "${LABEL},rank=${rank}" \
    -o jsonpath='{.items[0].metadata.name}'
}

extract_training_metrics() {
  local pod="$1"
  local logs
  logs="$(kubectl logs -n "${NAMESPACE}" "${pod}" --tail=200 2>/dev/null || true)"

  local iter="n/a"
  local loss="n/a"
  local throughput="n/a"

  local line
  while IFS= read -r line; do
    if [[ "${line}" =~ [Ii]ter(ation)?[^0-9]*([0-9]+) ]]; then
      iter="${BASH_REMATCH[2]}"
    fi
    if [[ "${line}" =~ [Ll]oss[^0-9]*([0-9]+\.?[0-9]*) ]]; then
      loss="${BASH_REMATCH[1]}"
    fi
    if [[ "${line}" =~ ([0-9]+\.?[0-9]*)[[:space:]]*(tok/s|tokens/s) ]]; then
      throughput="${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
    elif [[ "${line}" =~ [Tt]hroughput[^0-9]*([0-9]+\.?[0-9]*) ]]; then
      throughput="${BASH_REMATCH[1]}"
    fi
  done <<<"${logs}"

  echo "iter=${iter} loss=${loss} throughput=${throughput}"
}

extract_all_training_metrics() {
  local pods=()
  mapfile -t pods < <(get_pods)

  if [[ "${#pods[@]}" -eq 0 ]]; then
    echo "No matching pods"
    return
  fi

  local pod
  for pod in "${pods[@]}"; do
    if [[ -z "${pod}" ]]; then
      continue
    fi
    local metrics
    metrics="$(extract_training_metrics "${pod}")"
    printf "%-40s %s\n" "${pod}" "${metrics}"
  done
}

render_dashboard() {
  clear
  banner
  echo "Namespace: ${NAMESPACE}"
  echo "Label:     ${LABEL}"
  echo "Refresh:   ${REFRESH_SECONDS}s"
  echo "Time:      $(date '+%Y-%m-%d %H:%M:%S')"
  echo

  echo -e "${GREEN}POD STATUS${NC}"
  kubectl get pods \
    -n "${NAMESPACE}" \
    -l "${LABEL}" \
    -o wide || true

  echo
  echo -e "${GREEN}TRAINING METRICS (tail logs)${NC}"
  extract_all_training_metrics

  echo
  echo -e "${BLUE}Commands:${NC}"
  echo "  logs by rank: $(basename "$0") logs <rank>"
  echo "  rank details: $(basename "$0") rank <rank>"
  echo "  tensorboard:  $(basename "$0") tensorboard"
  echo "  stop dashboard: Ctrl+C"
}

mode_once() {
  banner
  kubectl get pods \
    -n "${NAMESPACE}" \
    -l "${LABEL}" \
    -o wide || true

  echo
  echo "Training metrics:"
  extract_all_training_metrics
}

mode_dashboard() {
  while true; do
    render_dashboard
    sleep "${REFRESH_SECONDS}"
  done
}

mode_logs() {
  local target="${1:-}"
  local pod=""

  if [[ -z "${target}" ]]; then
    mapfile -t pods < <(get_pods)
    if [[ "${#pods[@]}" -eq 0 ]]; then
      err "No pods found for ${LABEL}"
      exit 1
    fi
    pod="${pods[0]}"
  elif [[ "${target}" =~ ^[0-9]+$ ]]; then
    pod="$(get_pod_for_rank "${target}")"
  else
    pod="${target}"
  fi

  if [[ -z "${pod}" ]]; then
    err "Could not resolve pod for target: ${target}"
    exit 1
  fi

  info "Streaming logs for pod=${pod}"
  kubectl logs -f -n "${NAMESPACE}" "${pod}"
}

mode_rank() {
  local rank="${1:-}"
  if [[ -z "${rank}" ]]; then
    err "mode 'rank' requires rank id"
    usage
    exit 1
  fi

  local pod
  pod="$(get_pod_for_rank "${rank}")"
  if [[ -z "${pod}" ]]; then
    err "No pod found for rank=${rank}"
    exit 1
  fi

  banner
  echo "Rank: ${rank}"
  echo "Pod:  ${pod}"
  echo

  kubectl get pod "${pod}" -n "${NAMESPACE}" -o wide

  echo
  echo "Recent logs (last 120 lines):"
  kubectl logs -n "${NAMESPACE}" "${pod}" --tail=120 || true
}

mode_tensorboard() {
  banner
  info "Delegating TensorBoard workflow to deploy script"
  "${script_dir}/deploy_asteroid.sh" --phase tensorboard
}

parse_args() {
  MODE="dashboard"

  if [[ $# -gt 0 ]]; then
    case "$1" in
      dashboard|once|logs|rank|tensorboard)
        MODE="$1"
        shift
        ;;
    esac
  fi

  POSITIONAL=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n|--namespace)
        NAMESPACE="$2"
        shift 2
        ;;
      -l|--label)
        LABEL="$2"
        shift 2
        ;;
      -r|--refresh)
        REFRESH_SECONDS="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  case "${MODE}" in
    dashboard)
      mode_dashboard
      ;;
    once)
      mode_once
      ;;
    logs)
      mode_logs "${POSITIONAL[0]:-}"
      ;;
    rank)
      mode_rank "${POSITIONAL[0]:-}"
      ;;
    tensorboard)
      mode_tensorboard
      ;;
    *)
      err "Unknown mode: ${MODE}"
      usage
      exit 1
      ;;
  esac
}

main "$@"
