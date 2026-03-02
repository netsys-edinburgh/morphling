#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/../deploy_asteroid"
VENV_DIR="${SCRIPT_DIR}/../.venv"
PROFILES_DIR="${SCRIPT_DIR}/../profiles"
GENERATED_DIR="${DEPLOY_DIR}/generated"
PLAN_FILE="${SCRIPT_DIR}/../hpp_plan.json"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"

IMAGE_NAME="baselines"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
export ANSIBLE_CONFIG="${DEPLOY_DIR}/ansible.cfg"

ANSIBLE_INVENTORY="${DEPLOY_DIR}/inventory.ini"
ANSIBLE_SECRETS="${DEPLOY_DIR}/secrets.yml"
VAULT_PASSWORD_FILE="${HOME}/.baselines_vault_pass"
ANSIBLE_COMMON_FLAGS="-i ${ANSIBLE_INVENTORY} -e @${ANSIBLE_SECRETS}"

NAMESPACE="${NAMESPACE:-default}"
MASTER_PORT="${MASTER_PORT:-29500}"

RUN_K3S=1
RUN_GPU=1
RUN_PROFILE=1
RUN_PLAN=1
RUN_BUILD=1
RUN_MANIFESTS=1
RUN_DEPLOY=1
RUN_MONITOR=0
RUN_TENSORBOARD=0
REDEPLOY=0
STATUS_ONLY=0
PHASE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
  echo -e "${GREEN}[deploy-asteroid]${NC} $*"
}

info() {
  echo -e "${BLUE}[deploy-asteroid]${NC} $*"
}

warn() {
  echo -e "${YELLOW}[deploy-asteroid]${NC} $*"
}

err() {
  echo -e "${RED}[deploy-asteroid]${NC} $*" >&2
}

header() {
  echo
  echo -e "${BLUE}====================================================${NC}"
  echo -e "${BLUE}BASELINES-ASTEROID | $*${NC}"
  echo -e "${BLUE}====================================================${NC}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --phase <name>        Run a single phase:
                        k3s|gpu|profile|plan|build|manifests|deploy|
                        monitor|tensorboard
  --skip-k3s            Skip K3s setup phase
  --skip-gpu            Skip GPU runtime phase
  --skip-profile        Skip profiling phase
  --skip-plan           Skip planning phase
  --skip-build          Skip image build/distribution phase
  --redeploy            Force delete running pods before apply
  --monitor             Run monitor phase after deploy
  --status              Show cluster/job status and exit
  -h, --help            Show this help message

Environment:
  IMAGE_TAG             Docker image tag (default: latest)
  NAMESPACE             Kubernetes namespace (default: default)
  MASTER_PORT           torch.distributed port (default: 29500)
EOF
}

check_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    err "Required command missing: ${cmd}"
    exit 1
  fi
}

check_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    err "Required file missing: ${path}"
    exit 1
  fi
}

check_prereqs() {
  header "PREREQUISITES"

  check_cmd docker
  check_cmd kubectl

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    err "Missing Python venv executable: ${VENV_DIR}/bin/python"
    err "Create the venv first (expected at baselines/.venv)."
    exit 1
  fi

  if [[ ! -x "${VENV_DIR}/bin/ansible-playbook" ]]; then
    err "Missing ansible-playbook: ${VENV_DIR}/bin/ansible-playbook"
    exit 1
  fi

  if [[ ! -x "${VENV_DIR}/bin/ansible" ]]; then
    err "Missing ansible ad-hoc binary: ${VENV_DIR}/bin/ansible"
    exit 1
  fi

  check_file "${VAULT_PASSWORD_FILE}"
  check_file "${ANSIBLE_INVENTORY}"
  check_file "${ANSIBLE_SECRETS}"

  mkdir -p "${PROFILES_DIR}"
  mkdir -p "${GENERATED_DIR}"

  info "KUBECONFIG=${KUBECONFIG}"
  info "ANSIBLE_CONFIG=${ANSIBLE_CONFIG}"
  info "IMAGE=${IMAGE_FULL}"
  info "NAMESPACE=${NAMESPACE}"
}

run_ansible_playbook() {
  local playbook="$1"
  shift || true

  local playbook_path="${DEPLOY_DIR}/${playbook}"
  if [[ ! -f "${playbook_path}" ]]; then
    err "Playbook not found: ${playbook_path}"
    exit 1
  fi

  info "Running playbook: ${playbook}"
  "${VENV_DIR}/bin/ansible-playbook" \
    -i "${ANSIBLE_INVENTORY}" \
    --vault-password-file "${VAULT_PASSWORD_FILE}" \
    -e "@${ANSIBLE_SECRETS}" \
    "${playbook_path}" \
    "$@"
}

run_ansible_adhoc() {
  local hosts="$1"
  local module="$2"
  local module_args="$3"

  info "Running ad-hoc: hosts=${hosts} module=${module}"
  "${VENV_DIR}/bin/ansible" "${hosts}" \
    -i "${ANSIBLE_INVENTORY}" \
    --vault-password-file "${VAULT_PASSWORD_FILE}" \
    -e "@${ANSIBLE_SECRETS}" \
    -m "${module}" \
    -a "${module_args}" \
    -b
}

show_status() {
  header "BASELINES-ASTEROID STATUS"
  info "Cluster nodes"
  kubectl get nodes -o wide || true

  echo
  info "Training pods (${LABEL:-app=baselines-asteroid})"
  kubectl get pods \
    -n "${NAMESPACE}" \
    -l app=baselines-asteroid \
    -o wide || true
}

phase_k3s() {
  header "PHASE: K3S"
  run_ansible_playbook "setup_k3s.yaml" -v
  kubectl get nodes -o wide
}

phase_gpu() {
  header "PHASE: GPU RUNTIME"

  info "Installing nvidia-container-toolkit on cluster nodes"
  run_ansible_adhoc \
    "cluster" \
    "apt" \
    "name=nvidia-container-toolkit state=present update_cache=yes"

  info "Deploying containerd NVIDIA runtime config"
  run_ansible_adhoc \
    "cluster" \
    "file" \
    "path=/var/lib/rancher/k3s/agent/etc/containerd state=directory mode=0755"
  run_ansible_adhoc \
    "cluster" \
    "copy" \
    "src=${DEPLOY_DIR}/containerd-nvidia.toml.tmpl dest=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl mode=0644"

  info "Restarting K3s services"
  run_ansible_adhoc "master" "systemd" "name=k3s state=restarted enabled=yes"
  run_ansible_adhoc "workers" "systemd" "name=k3s-agent state=restarted enabled=yes"

  info "Deploying NVIDIA device plugin"
  kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml"
  kubectl -n kube-system rollout status \
    daemonset/nvidia-device-plugin-daemonset \
    --timeout=180s || true
}

phase_profile() {
  header "PHASE: PROFILE"
  run_ansible_playbook "profile_and_gather.yaml" -v
}

phase_plan() {
  header "PHASE: PLAN"

  local cluster_conf="${SCRIPT_DIR}/../cluster.conf"
  if [[ ! -f "${cluster_conf}" ]]; then
    err "Missing cluster.conf: ${cluster_conf}"
    exit 1
  fi

  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/run_asteroid_planner.py" \
    --profiles-dir "${PROFILES_DIR}" \
    --cluster-conf "${cluster_conf}" \
    --output "${PLAN_FILE}"
}

phase_build() {
  header "PHASE: BUILD"

  info "Common flags: ${ANSIBLE_COMMON_FLAGS}"
  run_ansible_playbook "distribute_image.yaml" -e "image_tag=${IMAGE_TAG}" -v
}

phase_manifests() {
  header "PHASE: MANIFESTS"

  if [[ ! -f "${PLAN_FILE}" ]]; then
    err "Plan file not found: ${PLAN_FILE}"
    err "Run plan phase first or provide ${PLAN_FILE}."
    exit 1
  fi

  mkdir -p "${GENERATED_DIR}"

  "${VENV_DIR}/bin/python" "${DEPLOY_DIR}/generate_manifests.py" \
    --plan "${PLAN_FILE}" \
    --image "${IMAGE_FULL}" \
    --namespace "${NAMESPACE}" \
    --master-port "${MASTER_PORT}" \
    --output-dir "${GENERATED_DIR}"
}

phase_deploy() {
  header "PHASE: DEPLOY"

  if [[ ! -d "${GENERATED_DIR}" ]]; then
    err "Manifest directory missing: ${GENERATED_DIR}"
    exit 1
  fi

  kubectl create namespace "${NAMESPACE}" >/dev/null 2>&1 || true

  info "Deleting old jobs with label app=baselines-asteroid"
  kubectl delete jobs \
    -n "${NAMESPACE}" \
    -l app=baselines-asteroid \
    --ignore-not-found=true || true

  if [[ "${REDEPLOY}" -eq 1 ]]; then
    info "Redeploy requested: deleting existing pods"
    kubectl delete pods \
      -n "${NAMESPACE}" \
      -l app=baselines-asteroid \
      --ignore-not-found=true || true
  fi

  info "Applying manifests from ${GENERATED_DIR}"
  kubectl apply -f "${GENERATED_DIR}/"

  info "Waiting for pods to become ready"
  kubectl wait --for=condition=Ready \
    pod \
    -l app=baselines-asteroid \
    -n "${NAMESPACE}" \
    --timeout=300s || warn "Timeout waiting for ready pods"

  kubectl get pods \
    -n "${NAMESPACE}" \
    -l app=baselines-asteroid \
    -o wide || true
}

phase_monitor() {
  header "PHASE: MONITOR"

  local monitor_script="${SCRIPT_DIR}/monitor_asteroid.sh"
  if [[ ! -x "${monitor_script}" ]]; then
    err "Monitor script missing or not executable: ${monitor_script}"
    exit 1
  fi

  "${monitor_script}" dashboard --namespace "${NAMESPACE}"
}

phase_tensorboard() {
  header "PHASE: TENSORBOARD"

  local tb_root="${SCRIPT_DIR}/../tensorboard"
  mkdir -p "${tb_root}"

  mapfile -t pods < <(
    kubectl get pods \
      -n "${NAMESPACE}" \
      -l app=baselines-asteroid \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'
  )

  if [[ "${#pods[@]}" -eq 0 ]]; then
    warn "No pods found for label app=baselines-asteroid"
    warn "Skipping TensorBoard sync"
    return 0
  fi

  info "Syncing TensorBoard logs from training pods"
  local pod
  for pod in "${pods[@]}"; do
    if [[ -z "${pod}" ]]; then
      continue
    fi
    local pod_dst="${tb_root}/${pod}"
    mkdir -p "${pod_dst}"
    kubectl cp "${NAMESPACE}/${pod}:/tensorboard/." "${pod_dst}" \
      >/dev/null 2>&1 || warn "Could not copy /tensorboard from ${pod}"
  done

  info "Launching TensorBoard"
  info "Logdir: ${tb_root}"
  info "URL: http://localhost:${TENSORBOARD_PORT:-6006}"

  if [[ -x "${VENV_DIR}/bin/tensorboard" ]]; then
    "${VENV_DIR}/bin/tensorboard" \
      --logdir "${tb_root}" \
      --host "${TENSORBOARD_HOST:-0.0.0.0}" \
      --port "${TENSORBOARD_PORT:-6006}"
  else
    "${VENV_DIR}/bin/python" -m tensorboard.main \
      --logdir "${tb_root}" \
      --host "${TENSORBOARD_HOST:-0.0.0.0}" \
      --port "${TENSORBOARD_PORT:-6006}"
  fi
}

run_phase_by_name() {
  case "$1" in
    k3s)
      phase_k3s
      ;;
    gpu)
      phase_gpu
      ;;
    profile)
      phase_profile
      ;;
    plan)
      phase_plan
      ;;
    build)
      phase_build
      ;;
    manifests)
      phase_manifests
      ;;
    deploy)
      phase_deploy
      ;;
    monitor)
      phase_monitor
      ;;
    tensorboard)
      phase_tensorboard
      ;;
    *)
      err "Unknown phase: $1"
      usage
      exit 1
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --phase)
        PHASE="$2"
        shift 2
        ;;
      --skip-k3s)
        RUN_K3S=0
        shift
        ;;
      --skip-gpu)
        RUN_GPU=0
        shift
        ;;
      --skip-profile)
        RUN_PROFILE=0
        shift
        ;;
      --skip-plan)
        RUN_PLAN=0
        shift
        ;;
      --skip-build)
        RUN_BUILD=0
        shift
        ;;
      --redeploy)
        REDEPLOY=1
        shift
        ;;
      --monitor)
        RUN_MONITOR=1
        shift
        ;;
      --status)
        STATUS_ONLY=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        err "Unknown argument: $1"
        usage
        exit 1
        ;;
    esac
  done
}

main() {
  parse_args "$@"

  if [[ "${STATUS_ONLY}" -eq 1 ]]; then
    show_status
    exit 0
  fi

  check_prereqs

  if [[ -n "${PHASE}" ]]; then
    run_phase_by_name "${PHASE}"
    exit 0
  fi

  if [[ "${RUN_K3S}" -eq 1 ]]; then
    phase_k3s
  fi

  if [[ "${RUN_GPU}" -eq 1 ]]; then
    phase_gpu
  fi

  if [[ "${RUN_PROFILE}" -eq 1 ]]; then
    phase_profile
  fi

  if [[ "${RUN_PLAN}" -eq 1 ]]; then
    phase_plan
  fi

  if [[ "${RUN_BUILD}" -eq 1 ]]; then
    phase_build
  fi

  if [[ "${RUN_MANIFESTS}" -eq 1 ]]; then
    phase_manifests
  fi

  if [[ "${RUN_DEPLOY}" -eq 1 ]]; then
    phase_deploy
  fi

  if [[ "${RUN_MONITOR}" -eq 1 ]]; then
    phase_monitor
  fi

  if [[ "${RUN_TENSORBOARD}" -eq 1 ]]; then
    phase_tensorboard
  fi

  header "BASELINES-ASTEROID DEPLOYMENT COMPLETE"
  info "Use monitor: ${SCRIPT_DIR}/monitor_asteroid.sh"
  info "Or run tensorboard: $(basename "$0") --phase tensorboard"
}

main "$@"
