#!/usr/bin/env bash
# ============================================================================
# Baselines-Asteroid — Full Deployment Script
# ============================================================================
# Single entry-point to deploy Asteroid distributed training on a K3s cluster.
#
# Usage:
#   ./deploy_asteroid.sh                     # Full deployment (all phases)
#   ./deploy_asteroid.sh --phase gpu         # Run only GPU setup phase
#   ./deploy_asteroid.sh --skip-build        # Skip Docker image rebuild
#   ./deploy_asteroid.sh --skip-profile      # Use existing profiles
#   ./deploy_asteroid.sh --redeploy          # Just redeploy K8s jobs (fastest)
#   ./deploy_asteroid.sh --monitor           # Deploy + open live monitor
#
# Prerequisites:
#   1. deploy_asteroid/inventory.ini configured (auto-generated from YAML)
#   2. deploy_asteroid/secrets.yml encrypted with ansible-vault
#   3. ~/.baselines_vault_pass contains vault password
#   4. Python venv at baselines/.venv/ with project dependencies
#
# Environment Variables:
#   IMAGE_TAG     - Docker image tag (default: latest)
#   NAMESPACE     - Kubernetes namespace (default: default)
#   MASTER_PORT   - torch.distributed port (default: 29500)
#   KUBECONFIG    - Path to kubeconfig (default: ~/.kube/config)
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}/../deploy_asteroid"
VENV_DIR="${SCRIPT_DIR}/../.venv"
PROFILES_DIR="${SCRIPT_DIR}/../profiles"
GENERATED_DIR="${DEPLOY_DIR}/generated"
PLAN_FILE="${SCRIPT_DIR}/../hpp_plan.json"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"

ASTEROID_CONFIG="${BASELINES_DIR}/configs/asteroid_default.yaml"
INVENTORY_TEMPLATE="${DEPLOY_DIR}/inventory.ini.template"

IMAGE_NAME="baselines"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

# Registry configuration
REGISTRY_PORT="${REGISTRY_PORT:-5000}"
REGISTRY_HOST=""  # auto-detected from master node IP
REGISTRY_IMAGE=""  # set after REGISTRY_HOST is resolved

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config}"
export ANSIBLE_CONFIG="${DEPLOY_DIR}/ansible.cfg"

ANSIBLE_INVENTORY="${DEPLOY_DIR}/inventory.ini"
ANSIBLE_SECRETS="${DEPLOY_DIR}/secrets.yml"
VAULT_PASSWORD_FILE="${HOME}/.baselines_vault_pass"
ANSIBLE_COMMON_FLAGS="-i ${ANSIBLE_INVENTORY} -e @${ANSIBLE_SECRETS}"

NAMESPACE="${NAMESPACE:-default}"
MASTER_PORT="${MASTER_PORT:-29500}"

APP_LABEL="baselines-asteroid"

# Phase control
RUN_K3S=1
RUN_REGISTRY=1
RUN_GPU=1
RUN_PROFILE=1
RUN_PLAN=1
RUN_BUILD=1
RUN_MANIFESTS=1
RUN_DEPLOY=1
RUN_MONITOR=0
RUN_TENSORBOARD=0
REDEPLOY=0
REDEPLOY_ONLY=0
STATUS_ONLY=0
PHASE=""
STRATEGY=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# Helpers
# ============================================================================
log()    { echo -e "${GREEN}[✓]${NC} $*"; }
info()   { echo -e "${BLUE}[ℹ]${NC} $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $*"; }
err()    { echo -e "${RED}[✗]${NC} $*" >&2; }
header() {
  echo ""
  echo -e "${CYAN}========================================${NC}"
  echo -e "${CYAN} $*${NC}"
  echo -e "${CYAN}========================================${NC}"
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

# ============================================================================
# YAML Config Helpers
# ============================================================================

# Read a single value from asteroid config YAML using Python
yaml_get() {
  local key_path="$1"
  "${VENV_DIR}/bin/python" - "${ASTEROID_CONFIG}" "${key_path}" <<'PYEOF'
import yaml, functools, operator, sys
config_path = sys.argv[1]
key_path = sys.argv[2]
with open(config_path) as f:
    d = yaml.safe_load(f)
keys = key_path.split('.')
try:
    val = functools.reduce(operator.getitem, keys, d)
    print(val if val is not None else '')
except (KeyError, TypeError):
    print('')
PYEOF
}

# Load config values from asteroid YAML (overrides env vars if YAML exists)
load_yaml_config() {
  if [[ ! -f "${ASTEROID_CONFIG}" ]]; then
    warn "Config not found at ${ASTEROID_CONFIG}, using defaults"
    return
  fi
  info "Loading config from ${ASTEROID_CONFIG}"

  local yaml_image_name
  yaml_image_name=$(yaml_get "deploy.image_name")
  [[ -n "$yaml_image_name" ]] && IMAGE_NAME="$yaml_image_name"

  local yaml_image_tag
  yaml_image_tag=$(yaml_get "deploy.image_tag")
  [[ -n "$yaml_image_tag" ]] && IMAGE_TAG="$yaml_image_tag"

  IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

  # Strategy (can be overridden by --strategy flag)
  if [[ -z "$STRATEGY" ]]; then
    STRATEGY=$(yaml_get "parallelism.strategy")
  fi
}

# ============================================================================
# Inventory Generation
# ============================================================================

generate_inventory() {
  # Auto-generate inventory.ini from asteroid_default.yaml cluster config
  local config_file="$1"
  local output_file="$2"

  info "Generating ${output_file} from ${config_file}"

  python3 - "${config_file}" "${output_file}" <<'PYEOF'
import yaml, sys, os

config_file = sys.argv[1]
output_file = sys.argv[2]

with open(config_file) as f:
    cfg = yaml.safe_load(f)

nodes = cfg.get('cluster', {}).get('nodes', [])
if not nodes:
    print('ERROR: No cluster.nodes found in config', file=sys.stderr)
    sys.exit(1)

master_nodes = [n for n in nodes if n.get('role') == 'master']
worker_nodes = [n for n in nodes if n.get('role') != 'master']

if not master_nodes:
    # Fall back: treat rank-0 as master
    master_nodes = [n for n in nodes if n.get('rank', -1) == 0]
    worker_nodes = [n for n in nodes if n.get('rank', -1) != 0]

lines = []
lines.append(f'# Auto-generated from {config_file}')
lines.append('# Do not edit — regenerate via deploy_asteroid.sh')
lines.append('')
lines.append('[master]')
lines.append('# Master node (Rank 0) — runs the rendezvous service')
for n in master_nodes:
    hname  = n.get('hostname', 'master')
    # Suffix with _node to avoid collision with [master] group name
    alias  = hname + '_node'
    ip     = n['ip']
    rank   = n.get('rank', 0)
    gpu_id = n.get('gpu_id', 0)
    nic    = n.get('nic', 'eth0')
    lines.append(
        f"{alias} ansible_host={ip} ansible_user='ubuntu' "
        f"rank={rank} gpu_id={gpu_id} nic={nic} hostname={hname}"
    )

lines.append('')
lines.append('[workers]')
lines.append('# Worker nodes')
for n in worker_nodes:
    hname  = n.get('hostname', f"worker{n.get('rank', 0)}")
    # Suffix with _node to avoid collision with group names
    alias  = hname + '_node'
    ip     = n['ip']
    rank   = n.get('rank', 0)
    gpu_id = n.get('gpu_id', 0)
    nic    = n.get('nic', 'eth0')
    lines.append(
        f"{alias} ansible_host={ip} ansible_user='ubuntu' "
        f"rank={rank} gpu_id={gpu_id} nic={nic} hostname={hname}"
    )

lines.append('')
lines.append('[all:vars]')
lines.append('ansible_python_interpreter=/usr/bin/python3')
lines.append('')
lines.append('[cluster:children]')
lines.append('master')
lines.append('workers')
lines.append('')
lines.append('[cluster:vars]')
lines.append('ansible_python_interpreter=/usr/bin/python3')
lines.append('baselines_venv=/opt/baselines/venv')
lines.append('baselines_src=/opt/baselines/src')
lines.append('')

with open(output_file, 'w') as f:
    f.write('\n'.join(lines))

print(f'Generated inventory with {len(master_nodes)} master(s) and {len(worker_nodes)} worker(s)')
PYEOF

  if [[ $? -ne 0 ]]; then
    err "Failed to generate inventory.ini"
    exit 1
  fi

  log "inventory.ini written to ${output_file}"
}

# ============================================================================
# Prerequisites
# ============================================================================

check_prereqs() {
  header "PREREQUISITES"

  local ok=true

  # Python venv
  if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    err "Python venv not found at ${VENV_DIR}"
    err "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
    ok=false
  fi

  # Ansible
  if [[ -f "${VENV_DIR}/bin/ansible-playbook" ]]; then
    : # ok
  elif command -v ansible-playbook &>/dev/null; then
    : # ok
  else
    err "ansible-playbook not found. Install: pip install ansible"
    ok=false
  fi

  # Docker
  if ! command -v docker &>/dev/null; then
    err "docker not found. Install Docker first."
    ok=false
  fi

  # kubectl
  if ! command -v kubectl &>/dev/null; then
    warn "kubectl not found locally — will rely on K3s nodes"
  fi

  # Vault password file
  if [[ ! -f "${VAULT_PASSWORD_FILE}" ]]; then
    err "Vault password file not found: ${VAULT_PASSWORD_FILE}"
    err "Create it: echo 'your-vault-password' > ${VAULT_PASSWORD_FILE} && chmod 600 ${VAULT_PASSWORD_FILE}"
    ok=false
  fi

  # Secrets file
  if [[ ! -f "${ANSIBLE_SECRETS}" ]]; then
    err "Secrets file not found: ${ANSIBLE_SECRETS}"
    err "Create it: cd deploy_asteroid && cp secrets.yml.template secrets.yml && ansible-vault encrypt secrets.yml"
    ok=false
  fi

  # Always regenerate inventory.ini from YAML config to stay in sync
  if [[ -f "${ASTEROID_CONFIG}" ]]; then
    info "Regenerating inventory.ini from ${ASTEROID_CONFIG}"
    generate_inventory "${ASTEROID_CONFIG}" "${ANSIBLE_INVENTORY}"
  else
    err "Cannot generate inventory: missing ${ASTEROID_CONFIG}"
    ok=false
  fi

  if [[ "$ok" == false ]]; then
    err "Prerequisites check failed. See errors above."
    exit 1
  fi

  mkdir -p "${PROFILES_DIR}"
  mkdir -p "${GENERATED_DIR}"

  log "Prerequisites OK"
  info "KUBECONFIG=${KUBECONFIG}"
  info "ANSIBLE_CONFIG=${ANSIBLE_CONFIG}"
  info "IMAGE=${IMAGE_FULL}"
  info "NAMESPACE=${NAMESPACE}"
}

# ============================================================================
# Ansible Wrappers
# ============================================================================

run_ansible_playbook() {
  local playbook="$1"
  shift || true

  local playbook_path="${DEPLOY_DIR}/${playbook}"
  if [[ ! -f "${playbook_path}" ]]; then
    err "Playbook not found: ${playbook_path}"
    exit 1
  fi

  info "Running playbook: $(basename "${playbook_path}")"

  # Use venv ansible if available, otherwise system
  local ansible_cmd="ansible-playbook"
  if [[ -f "${VENV_DIR}/bin/ansible-playbook" ]]; then
    ansible_cmd="${VENV_DIR}/bin/ansible-playbook"
  fi

  ${ansible_cmd} \
    -i "${ANSIBLE_INVENTORY}" \
    --vault-password-file "${VAULT_PASSWORD_FILE}" \
    -e "@${ANSIBLE_SECRETS}" \
    "${playbook_path}" \
    "$@"
}

run_ansible_adhoc() {
  # Run ad-hoc ansible command using vault secrets (no -k/-K needed)
  local hosts="$1"
  shift

  local ansible_cmd="ansible"
  if [[ -f "${VENV_DIR}/bin/ansible" ]]; then
    ansible_cmd="${VENV_DIR}/bin/ansible"
  fi

  ${ansible_cmd} "${hosts}" \
    -i "${ANSIBLE_INVENTORY}" \
    --vault-password-file "${VAULT_PASSWORD_FILE}" \
    -e "@${ANSIBLE_SECRETS}" \
    "$@"
}

# ============================================================================
# Phase Functions
# ============================================================================

phase_k3s() {
  header "Phase 1: K3s Cluster Setup"

  # Check if K3s is already running
  if kubectl get nodes &>/dev/null 2>&1; then
    local node_count
    node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    log "K3s cluster already running with ${node_count} node(s)"
    kubectl get nodes
    return 0
  fi

  info "Installing K3s cluster..."
  run_ansible_playbook "setup_k3s.yaml" -v
  log "K3s cluster setup complete"

  # Verify
  sleep 5
  kubectl get nodes -o wide
}

# --------------------------------------------------------------------------
# Resolve the master (rank-0) node IP from config YAML for registry address
# --------------------------------------------------------------------------
resolve_registry_host() {
  if [[ -n "${REGISTRY_HOST}" ]]; then
    return 0
  fi
  REGISTRY_HOST=$(
    python3 - "${ASTEROID_CONFIG}" <<'PYEOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
nodes = cfg.get('cluster', {}).get('nodes', [])
master = [n for n in nodes if n.get('role') == 'master']
if not master:
    master = [n for n in nodes if n.get('rank', -1) == 0]
if master:
    print(master[0]['ip'])
else:
    print('')
PYEOF
  )
  if [[ -z "${REGISTRY_HOST}" ]]; then
    err "Cannot determine master node IP for registry"
    exit 1
  fi
  REGISTRY_IMAGE="${REGISTRY_HOST}:${REGISTRY_PORT}/${IMAGE_NAME}:${IMAGE_TAG}"
  log "Registry address: ${REGISTRY_HOST}:${REGISTRY_PORT}"
}

phase_registry() {
  header "Phase 1b: Private Docker Registry"

  resolve_registry_host
  local reg_addr="${REGISTRY_HOST}:${REGISTRY_PORT}"

  # ── Step 1: Start registry container on the master node ─────────────
  info "Ensuring Docker registry on master (${reg_addr})..."
  run_ansible_adhoc "master" -m shell \
    -a "docker ps -q --filter name='^baselines-registry$' | grep -q . \
         && echo RUNNING \
         || docker run -d --restart=always \
              --name baselines-registry \
              -p ${REGISTRY_PORT}:5000 \
              -v /opt/baselines/registry:/var/lib/registry \
              registry:2" \
    --become
  log "Registry container running on master"

  # ── Step 2: Health check ────────────────────────────────────────────
  info "Waiting for registry health check..."
  local retries=10
  for i in $(seq 1 $retries); do
    if run_ansible_adhoc "master" -m uri \
        -a "url=http://localhost:${REGISTRY_PORT}/v2/ return_content=yes" \
        --become &>/dev/null; then
      log "Registry healthy"
      break
    fi
    if [[ $i -eq $retries ]]; then
      err "Registry not responding after ${retries} attempts"
      exit 1
    fi
    sleep 2
  done

  # ── Step 3: Configure K3s containerd to trust the private registry ──
  #   Creates /etc/rancher/k3s/registries.yaml on every node so that
  #   containerd pulls from our insecure registry without TLS.
  info "Configuring K3s containerd to use registry (${reg_addr})..."
  local registries_yaml
  registries_yaml=$(cat <<REGYAML
mirrors:
  "${reg_addr}":
    endpoint:
      - "http://${reg_addr}"
configs:
  "${reg_addr}":
    tls:
      insecure_skip_verify: true
REGYAML
  )
  # Ensure the directory exists on all nodes before writing config
  run_ansible_adhoc "cluster" -m file \
    -a "path=/etc/rancher/k3s state=directory mode=0755" \
    --become
  run_ansible_adhoc "cluster" -m copy \
    -a "content='${registries_yaml}' dest=/etc/rancher/k3s/registries.yaml mode=0644" \
    --become
  log "registries.yaml deployed to all nodes"

  # ── Step 4: Restart K3s so containerd picks up new registry config ──
  info "Restarting K3s services to apply registry config..."
  run_ansible_adhoc "cluster" -m shell \
    -a "systemctl restart k3s 2>/dev/null || systemctl restart k3s-agent 2>/dev/null || echo 'no k3s service found (will apply on next start)'" \
    --become
  sleep 10

  # Wait for nodes to be Ready
  local retries=12
  for i in $(seq 1 $retries); do
    if ! kubectl get nodes 2>/dev/null | grep -q "NotReady"; then
      log "All nodes Ready after registry config"
      break
    fi
    if [[ $i -eq $retries ]]; then
      err "Nodes did not become Ready after restart"
      kubectl get nodes
      exit 1
    fi
    info "Waiting for nodes... (${i}/${retries})"
    sleep 10
  done

  # ── Step 5: Ensure local Docker can push to the registry ────────────
  #   The build machine (control node) also needs to be able to push
  info "Configuring local Docker daemon for insecure registry..."
  local daemon_json="/etc/docker/daemon.json"
  if [[ -f "${daemon_json}" ]]; then
    # Add our registry to existing insecure-registries if not already there
    if ! grep -q "${reg_addr}" "${daemon_json}" 2>/dev/null; then
      python3 - "${daemon_json}" "${reg_addr}" <<'PYEOF'
import json, sys
path, addr = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        cfg = json.load(f)
except (json.JSONDecodeError, FileNotFoundError):
    cfg = {}
cfg.setdefault('insecure-registries', [])
if addr not in cfg['insecure-registries']:
    cfg['insecure-registries'].append(addr)
with open(path, 'w') as f:
    json.dump(cfg, f, indent=2)
PYEOF
      sudo systemctl restart docker
      sleep 3
      log "Local Docker restarted with insecure registry ${reg_addr}"
    else
      log "Local Docker already configured for ${reg_addr}"
    fi
  else
    echo "{\"insecure-registries\": [\"${reg_addr}\"]}" | sudo tee "${daemon_json}" > /dev/null
    sudo systemctl restart docker
    sleep 3
    log "Local Docker configured for insecure registry ${reg_addr}"
  fi

  log "Private registry ready at ${reg_addr}"
}

phase_gpu() {
  header "Phase 2: GPU & NVIDIA Runtime Setup"

  # Check if NVIDIA device plugin is already running
  if kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds --no-headers 2>/dev/null | grep -q Running; then
    local gpu_count
    gpu_count=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null | grep -c "1" || true)
    if [[ "$gpu_count" -gt 0 ]]; then
      log "NVIDIA device plugin running, ${gpu_count} GPU(s) available"
      return 0
    fi
  fi

  # Prefer Ansible playbook if it exists (matches asteroid_project)
  if [[ -f "${DEPLOY_DIR}/setup_gpu.yaml" ]]; then
    info "Running setup_gpu.yaml playbook..."
    run_ansible_playbook setup_gpu.yaml -v
    log "GPU setup via playbook complete"
    return 0
  fi

  # Fallback: inline ad-hoc commands
  # Step 1: Install nvidia-container-toolkit on all nodes
  info "Installing nvidia-container-toolkit..."
  run_ansible_adhoc "cluster" -m shell \
    -a "apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit 2>/dev/null || echo 'toolkit already installed or repo missing'" \
    --become

  # Step 2: Deploy containerd config.toml.tmpl
  info "Deploying containerd nvidia config template..."
  run_ansible_adhoc "cluster" -m file \
    -a "path=/var/lib/rancher/k3s/agent/etc/containerd state=directory mode=0755" \
    --become
  run_ansible_adhoc "cluster" -m copy \
    -a "src=${DEPLOY_DIR}/containerd-nvidia.toml.tmpl dest=/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl mode=0644" \
    --become

  # Step 3: Restart K3s services
  info "Restarting K3s services..."
  run_ansible_adhoc "master" -m systemd -a "name=k3s state=restarted" --become
  run_ansible_adhoc "workers" -m systemd -a "name=k3s-agent state=restarted" --become

  info "Waiting for nodes to recover..."
  sleep 15

  # Step 4: Wait for all nodes to be Ready
  local retries=12
  for i in $(seq 1 $retries); do
    if ! kubectl get nodes 2>/dev/null | grep -q "NotReady"; then
      log "All nodes Ready"
      break
    fi
    if [[ $i -eq $retries ]]; then
      err "Nodes did not become Ready after restart"
      kubectl get nodes
      exit 1
    fi
    info "Waiting for nodes... (${i}/${retries})"
    sleep 10
  done

  # Step 5: Deploy NVIDIA device plugin
  info "Deploying NVIDIA device plugin..."
  kubectl apply -f "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.2/deployments/static/nvidia-device-plugin.yml"
  kubectl -n kube-system rollout status \
    daemonset/nvidia-device-plugin-daemonset \
    --timeout=180s || true
  sleep 10

  # Step 6: Verify GPUs
  info "Checking GPU resources..."
  kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu"

  log "GPU setup complete"
}

phase_profile() {
  header "Phase 3: Cluster Profiling"

  mkdir -p "${PROFILES_DIR}"

  local profile_count
  profile_count=$(find "${PROFILES_DIR}" -name "profile_*.json" 2>/dev/null | wc -l)

  if [[ "$profile_count" -gt 0 ]]; then
    log "Found ${profile_count} existing profile(s), skipping profiling"
    ls -la "${PROFILES_DIR}"/profile_*.json
    return 0
  fi

  info "Running cluster profiling..."
  run_ansible_playbook "profile_and_gather.yaml" -v

  profile_count=$(find "${PROFILES_DIR}" -name "profile_*.json" 2>/dev/null | wc -l)
  log "Profiling complete: ${profile_count} profile(s) collected"
}

phase_plan() {
  header "Phase 4: HPP Planning"

  if [[ -f "${PLAN_FILE}" ]]; then
    log "Existing plan found: ${PLAN_FILE}"
    "${VENV_DIR}/bin/python" - "${PLAN_FILE}" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    p = json.load(f)
print(f'  Stages:    {p.get("num_stages", "?")}')
print(f'  World:     {p.get("world_size", "?")}')
print(f'  Partition: {p.get("partition_points", "?")}')
lat = p.get("estimated_latency_ms")
print(f'  Latency:   {lat:.1f} ms' if lat else '  Latency:   ?')
PYEOF
    return 0
  fi

  info "Running HPP planner..."
  local planner_args=(
    --profiles-dir "${PROFILES_DIR}"
    --output "${PLAN_FILE}"
  )
  # Prefer asteroid.yaml for cluster info; fall back to cluster.conf
  if [[ -f "${ASTEROID_CONFIG}" ]]; then
    planner_args+=(--config "${ASTEROID_CONFIG}")
  else
    local cluster_conf="${BASELINES_DIR}/cluster.conf"
    if [[ ! -f "${cluster_conf}" ]]; then
      err "Missing cluster.conf: ${cluster_conf}"
      exit 1
    fi
    planner_args+=(--cluster-conf "${cluster_conf}")
  fi
  [[ -n "$STRATEGY" ]] && planner_args+=(--strategy "$STRATEGY")

  "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/run_asteroid_planner.py" "${planner_args[@]}"

  log "Plan generated: ${PLAN_FILE}"
}

phase_build() {
  header "Phase 5: Build & Push Docker Image"

  resolve_registry_host
  local reg_addr="${REGISTRY_HOST}:${REGISTRY_PORT}"

  # Step 0: Generate requirements.txt from pyproject.toml
  info "Generating requirements.txt from pyproject.toml..."
  python3 - "${BASELINES_DIR}/pyproject.toml" "${BASELINES_DIR}/requirements.txt" <<'PYEOF'
import re, ast, sys

toml_path, out_path = sys.argv[1], sys.argv[2]
text = open(toml_path).read()
m = re.search(r'dependencies\s*=\s*(\[.*?\])', text, re.DOTALL)
if not m:
    print("ERROR: No dependencies found in pyproject.toml", file=sys.stderr)
    sys.exit(1)
deps = ast.literal_eval(m.group(1))
with open(out_path, 'w') as f:
    f.write("# Auto-generated from pyproject.toml by deploy_asteroid.sh\n")
    f.write("# Do not edit manually — update pyproject.toml instead.\n")
    for dep in deps:
        f.write(dep + "\n")
print(f"Generated {out_path} with {len(deps)} dependencies")
PYEOF
  log "requirements.txt generated"

  # Step 1: Build the Docker image locally
  info "Building Docker image: ${IMAGE_FULL}"
  docker build -t "${IMAGE_FULL}" -f "${BASELINES_DIR}/Dockerfile" "${BASELINES_DIR}"
  log "Image built: ${IMAGE_FULL}"

  # Step 2: Tag for registry and push
  #   - Single push to registry (only changed layers are uploaded)
  #   - All K3s nodes pull in parallel from the registry when pods start
  #   - Layer-level deduplication: subsequent deploys push only changed layers
  info "Tagging image for registry: ${REGISTRY_IMAGE}"
  docker tag "${IMAGE_FULL}" "${REGISTRY_IMAGE}"

  info "Pushing image to private registry (${reg_addr})..."
  local push_start push_end push_sec
  push_start=$(date +%s)
  docker push "${REGISTRY_IMAGE}"
  push_end=$(date +%s)
  push_sec=$(( push_end - push_start ))
  log "Image pushed to registry in ${push_sec}s"

  # Step 3: Remove old images from K3s containerd cache (fire-and-forget)
  info "Removing stale cached images on all nodes (async)..."
  run_ansible_adhoc "cluster" -m shell \
    -a "k3s ctr images rm ${REGISTRY_IMAGE} 2>/dev/null || true" \
    --become \
    -B 300 -P 0
  sleep 2

  # Step 4: Pre-pull image on all nodes in parallel (optional but speeds up pod start)
  #   Uses crictl pull which goes through containerd → registry.
  #   Each node pulls only the layers it's missing (layer caching).
  info "Pre-pulling image on all nodes (parallel, async)..."
  run_ansible_adhoc "cluster" -m shell \
    -a "k3s crictl pull ${REGISTRY_IMAGE}" \
    --become \
    -B 600 -P 10

  # Step 5: Verify image is available on all nodes
  info "Verifying image on all nodes..."
  run_ansible_adhoc "cluster" -m shell \
    -a "k3s crictl images | grep ${IMAGE_NAME} | head -3" \
    --become

  log "Image available on all nodes via registry (${reg_addr})"
}

phase_manifests() {
  header "Phase 6: Generate K8s Manifests"

  if [[ ! -f "${PLAN_FILE}" ]]; then
    err "Plan file not found: ${PLAN_FILE}"
    err "Run plan phase first or provide ${PLAN_FILE}."
    exit 1
  fi

  mkdir -p "${GENERATED_DIR}"

  # Use registry image reference if registry is configured
  resolve_registry_host
  local manifest_image="${REGISTRY_IMAGE:-${IMAGE_FULL}}"

  info "Generating manifests from HPP plan (image: ${manifest_image})..."
  "${VENV_DIR}/bin/python" "${DEPLOY_DIR}/generate_manifests.py" \
    --plan "${PLAN_FILE}" \
    --image "${manifest_image}" \
    --namespace "${NAMESPACE}" \
    --master-port "${MASTER_PORT}" \
    --image-pull-policy "Always" \
    --output-dir "${GENERATED_DIR}"

  # Verify NCCL_SOCKET_IFNAME is set to eth0 in manifests (K8s pods use eth0)
  info "Verifying NCCL_SOCKET_IFNAME is set to eth0 in manifests..."
  local bad_ifname
  bad_ifname=$(grep -l 'value: "ens' "${GENERATED_DIR}"/02-job-rank-*.yaml 2>/dev/null || true)
  if [[ -n "$bad_ifname" ]]; then
    warn "Found host NIC name in manifests, fixing to eth0..."
    sed -i 's/value: "ens[0-9]*"/value: "eth0"/g' "${GENERATED_DIR}"/02-job-rank-*.yaml
  fi

  log "Manifests generated in ${GENERATED_DIR}/"
  ls -la "${GENERATED_DIR}"/*.yaml 2>/dev/null || true
}

phase_deploy() {
  header "Phase 7: Deploy to K8s"

  if [[ ! -d "${GENERATED_DIR}" ]]; then
    err "Manifest directory missing: ${GENERATED_DIR}"
    exit 1
  fi

  kubectl create namespace "${NAMESPACE}" >/dev/null 2>&1 || true

  # Delete existing jobs
  info "Cleaning up old deployments..."
  kubectl delete jobs \
    -n "${NAMESPACE}" \
    -l "app=${APP_LABEL}" \
    --ignore-not-found=true || true
  sleep 3

  if [[ "${REDEPLOY}" -eq 1 ]]; then
    info "Redeploy requested: deleting existing pods"
    kubectl delete pods \
      -n "${NAMESPACE}" \
      -l "app=${APP_LABEL}" \
      --ignore-not-found=true || true
  fi

  # Apply all manifests
  info "Applying manifests from ${GENERATED_DIR}"
  kubectl apply -f "${GENERATED_DIR}/"
  log "Manifests applied"

  # Wait for pods
  info "Waiting for pods to start..."
  sleep 10

  local retries=30
  for i in $(seq 1 $retries); do
    local running
    running=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --no-headers 2>/dev/null | grep -c "Running" || true)
    local total
    total=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --no-headers 2>/dev/null | wc -l)

    if [[ "$running" -eq "$total" && "$total" -gt 0 ]]; then
      log "All ${total} pods running"
      break
    fi

    if [[ $i -eq $retries ]]; then
      err "Pods did not all reach Running state"
      kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" -o wide
      echo ""
      err "Check logs: kubectl logs -n ${NAMESPACE} -l app=${APP_LABEL} --tail=20"
      exit 1
    fi

    info "Pods: ${running}/${total} running (waiting ${i}/${retries})..."
    sleep 5
  done

  echo ""
  kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" -o wide
  echo ""

  # Quick check for training output
  info "Checking for training output (waiting 30s)..."
  sleep 30

  local last_rank_pod
  last_rank_pod=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --sort-by=.metadata.name --no-headers | tail -1 | awk '{print $1}')
  local training_line
  training_line=$(kubectl logs -n "${NAMESPACE}" "${last_rank_pod}" --tail=5 2>/dev/null | grep "iter" | tail -1 || true)

  if [[ -n "$training_line" ]]; then
    log "Training is running!"
    echo "  ${training_line}"
  else
    warn "No training output yet — pods may still be initializing"
    info "Check manually: kubectl logs -n ${NAMESPACE} ${last_rank_pod} --tail=20"
  fi
}

phase_monitor() {
  header "Phase 8: Training Monitor"

  local monitor_script="${SCRIPT_DIR}/monitor_asteroid.sh"
  if [[ ! -x "${monitor_script}" ]]; then
    err "Monitor script missing or not executable: ${monitor_script}"
    exit 1
  fi

  "${monitor_script}" dashboard --namespace "${NAMESPACE}"
}

phase_tensorboard() {
  header "Phase 9: TensorBoard Dashboard"

  # Verify training pods exist
  local pod_count
  pod_count=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --no-headers 2>/dev/null | wc -l)
  if [[ "$pod_count" -eq 0 ]]; then
    err "No training pods found. Deploy first."
    exit 1
  fi

  local tb_root="${BASELINES_DIR}/tb_logs"
  mkdir -p "${tb_root}"

  # Sync TB logs from ALL pods (each rank writes to its own subdir)
  info "Syncing TensorBoard logs from all ${pod_count} pods..."

  local pods
  pods=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --no-headers -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName 2>/dev/null)

  while IFS= read -r line; do
    local pod_name node_name
    pod_name=$(echo "$line" | awk '{print $1}')
    node_name=$(echo "$line" | awk '{print $2}')
    local rank
    rank=$(echo "$pod_name" | grep -oP 'rank-\K\d+' || echo "?")

    info "  Syncing rank-${rank} from ${node_name} (${pod_name})..."
    mkdir -p "${tb_root}/rank-${rank}"
    kubectl cp -n "${NAMESPACE}" "${pod_name}:/tensorboard/rank-${rank}/" "${tb_root}/rank-${rank}/" 2>/dev/null || {
      kubectl cp -n "${NAMESPACE}" "${pod_name}:/tensorboard/" "${tb_root}/rank-${rank}/" 2>/dev/null || true
    }
  done <<< "$pods"

  # Check if we got any TB data
  local event_files
  event_files=$(find "${tb_root}" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
  if [[ "$event_files" -eq 0 ]]; then
    warn "No TensorBoard event files found yet."
    info "Training may not have started writing logs."
    info "Try again later: $(basename "$0") --phase tensorboard"
    return 0
  fi

  log "Found ${event_files} event file(s) across $(find "${tb_root}" -mindepth 1 -maxdepth 1 -type d | wc -l) rank(s)"
  find "${tb_root}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r d; do
    local count
    count=$(find "$d" -name "events.out.tfevents.*" | wc -l)
    echo "  $(basename "$d"): ${count} event file(s)"
  done

  # Launch TensorBoard
  local tb_cmd="tensorboard"
  if [[ -f "${VENV_DIR}/bin/tensorboard" ]]; then
    tb_cmd="${VENV_DIR}/bin/tensorboard"
  fi

  info "Starting TensorBoard on http://localhost:${TENSORBOARD_PORT:-6006}"
  info "Each rank appears as a separate run in TensorBoard"
  info "Press Ctrl+C to stop"
  echo ""
  ${tb_cmd} --logdir "${tb_root}" --host "${TENSORBOARD_HOST:-0.0.0.0}" --port "${TENSORBOARD_PORT:-6006}"
}

phase_mps() {
  header "Phase 10: MPS Setup"

  local mps_enabled
  mps_enabled=$(yaml_get "mps.enabled")
  if [[ "$mps_enabled" != "True" && "$mps_enabled" != "true" ]]; then
    info "MPS is disabled in config — skipping"
    return 0
  fi

  local thread_pct
  thread_pct=$(yaml_get "mps.active_thread_percentage")
  [[ -z "$thread_pct" ]] && thread_pct=50

  info "Starting NVIDIA MPS on all nodes (active_thread_percentage=${thread_pct})..."

  run_ansible_adhoc "cluster" -m shell \
    -a "nvidia-smi -i 0 -c EXCLUSIVE_PROCESS 2>/dev/null; \
        nvidia-cuda-mps-control -d 2>/dev/null || true; \
        echo 'set_active_thread_percentage ${thread_pct}' | nvidia-cuda-mps-control 2>/dev/null || true" \
    --become

  info "Verifying MPS on all nodes..."
  run_ansible_adhoc "cluster" -m shell \
    -a "echo 'get_server_list' | nvidia-cuda-mps-control 2>/dev/null && echo 'MPS running' || echo 'MPS not running'" \
    --become

  log "MPS setup complete (${thread_pct}% active threads)"
}

# ============================================================================
# Checkpoint Collection & Merging
# ============================================================================

collect_checkpoints() {
  local dest="${1:-./checkpoints_collected}"
  mkdir -p "$dest"

  header "Collecting Checkpoints from Cluster Nodes"

  # Read cluster node IPs from config
  local -a ips hostnames
  eval "$("${VENV_DIR}/bin/python" - "${ASTEROID_CONFIG}" <<'PYEOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
nodes = cfg.get('cluster', {}).get('nodes', [])
ips = ' '.join(n['ip'] for n in nodes)
hostnames = ' '.join(n.get('hostname', 'unknown') for n in nodes)
print(f'ips=({ips})')
print(f'hostnames=({hostnames})')
PYEOF
  )"

  if [[ ${#ips[@]} -eq 0 ]]; then
    err "No cluster nodes found in config"
    return 1
  fi

  local total=0
  local collected_ranks=""

  for i in "${!ips[@]}"; do
    local ip="${ips[$i]}"
    local hostname="${hostnames[$i]}"

    log "Scanning ${hostname} (${ip}) for checkpoints..."

    local rank_dirs
    rank_dirs=$(ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ip" \
      "ls -d /var/lib/baselines/checkpoints/rank-* 2>/dev/null" 2>/dev/null || true)

    if [[ -z "$rank_dirs" ]]; then
      log "  No checkpoint directories found on ${hostname}"
      continue
    fi

    for src in $rank_dirs; do
      local rank_name
      rank_name=$(basename "$src")
      local rank_num=${rank_name#rank-}

      # Skip if already collected from another node
      if [[ "$collected_ranks" == *":${rank_num}:"* ]]; then
        log "  Skipping ${rank_name} (already collected from another node)"
        continue
      fi

      local count
      count=$(ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ip" \
        "ls ${src}/*.pt 2>/dev/null | wc -l" 2>/dev/null || echo "0")
      count=$(echo "$count" | tr -d '[:space:]')

      if [[ "$count" -gt 0 ]]; then
        local rank_dir="${dest}/${rank_name}"
        mkdir -p "$rank_dir"
        scp -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
          "${ip}:${src}/*.pt" "$rank_dir/" 2>/dev/null
        log "  Copied ${count} checkpoint(s) from ${rank_name}"
        total=$((total + count))
        collected_ranks="${collected_ranks}:${rank_num}:"
      else
        log "  ${rank_name}: empty (no .pt files)"
      fi
    done
  done

  echo ""
  if [[ "$total" -gt 0 ]]; then
    log "Collected ${total} checkpoint file(s) to ${dest}/"
    echo ""
    echo "Checkpoint files:"
    find "$dest" -name "*.pt" -printf "  %p (%s bytes)\n" | sort
  else
    warn "No checkpoint files found on any node."
    warn "Training may not have saved checkpoints (check checkpoint_interval in config)."
  fi
}

merge_checkpoints() {
  local checkpoint_dir="${1:-./checkpoints_collected}"
  local iteration="${2:-}"

  header "Merging Distributed Checkpoints"

  if [[ ! -d "$checkpoint_dir" ]]; then
    err "Checkpoint directory not found: $checkpoint_dir"
    err "Run '$(basename "$0") --checkpoints' first to collect checkpoints"
    return 1
  fi

  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi

  local cmd="${VENV_DIR}/bin/python -m baselines.ft.merge_checkpoints -d ${checkpoint_dir}"
  if [[ -n "$iteration" ]]; then
    cmd="${cmd} -i ${iteration}"
  fi

  log "Running: ${cmd}"
  echo ""
  eval "$cmd"
}

# ============================================================================
# Stop & Clean
# ============================================================================

stop_and_clean() {
  header "Deep Stop & Clean — All Nodes"

  # 1. Kill all K8s resources
  info "Deleting ALL ${APP_LABEL} K8s resources (jobs, pods, services, configmaps)..."

  kubectl delete jobs -n "${NAMESPACE}" -l "app=${APP_LABEL}" --ignore-not-found=true --force --grace-period=0 2>/dev/null || true
  kubectl delete pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --ignore-not-found=true --force --grace-period=0 2>/dev/null || true

  # Catch stuck pods
  local stuck_pods
  stuck_pods=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -i "asteroid\|rank-" | awk '{print $1}' || true)
  if [[ -n "$stuck_pods" ]]; then
    info "Force-deleting stuck pods: ${stuck_pods}"
    echo "$stuck_pods" | xargs kubectl delete pod -n "${NAMESPACE}" --force --grace-period=0 2>/dev/null || true
  fi

  kubectl delete service -n "${NAMESPACE}" "${APP_LABEL}-headless" --ignore-not-found=true 2>/dev/null || true
  kubectl delete configmap -n "${NAMESPACE}" "${APP_LABEL}-plan" --ignore-not-found=true 2>/dev/null || true

  for kind in deployment statefulset daemonset replicaset cronjob; do
    kubectl delete "$kind" -n "${NAMESPACE}" -l "app=${APP_LABEL}" --ignore-not-found=true 2>/dev/null || true
  done
  log "K8s resources deleted"

  # 2. Wait for pods to terminate
  info "Waiting for all pods to terminate..."
  local retries=12
  for i in $(seq 1 $retries); do
    local remaining
    remaining=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | grep -ci "asteroid\|rank-" || true)
    if [[ "$remaining" -eq 0 ]]; then
      log "All pods terminated"
      break
    fi
    if [[ $i -eq $retries ]]; then
      warn "${remaining} pod(s) still lingering — they will be cleaned by kubelet"
    fi
    sleep 5
  done

  # 3. Kill GPU processes on ALL cluster nodes
  info "Killing GPU processes on all cluster nodes via SSH..."

  local node_ips
  node_ips=$("${VENV_DIR}/bin/python" - "${ASTEROID_CONFIG}" <<'PYEOF'
import yaml, sys
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
for n in cfg.get('cluster', {}).get('nodes', []):
    print(n['ip'])
PYEOF
  ) 2>/dev/null || true

  if [[ -n "$node_ips" ]]; then
    while IFS= read -r ip; do
      info "  Cleaning GPU processes on ${ip}..."
      ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "ubuntu@${ip}" bash -s 2>/dev/null <<'REMOTE_CLEAN'
# Kill any python/pytorch GPU processes (training workers)
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n "$GPU_PIDS" ]; then
    echo "  Killing GPU PIDs: $GPU_PIDS"
    echo "$GPU_PIDS" | xargs -r kill -9 2>/dev/null || true
else
    echo "  No GPU processes found"
fi
pkill -9 -f "worker.py" 2>/dev/null || true
pkill -9 -f "profile_node.py" 2>/dev/null || true
pkill -9 -f "run_planner.py" 2>/dev/null || true
rm -f /dev/shm/nccl-* /dev/shm/torch_* 2>/dev/null || true
REMOTE_CLEAN
      log "  ${ip} cleaned"
    done <<< "$node_ips"
  else
    warn "Could not read node IPs from config — skipping remote GPU cleanup"
  fi

  # 4. Clean checkpoint data on all nodes
  info "Cleaning checkpoint data on all nodes..."
  if [[ -n "$node_ips" ]]; then
    while IFS= read -r ip; do
      ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "ubuntu@${ip}" \
        "sudo rm -rf /var/lib/baselines/checkpoints/rank-*/*.pt 2>/dev/null; echo 'checkpoints cleared'" \
        2>/dev/null || true
    done <<< "$node_ips"
    log "Remote checkpoints cleared"
  fi

  # 5. Clean local artifacts
  info "Removing local generated files..."

  if [[ -d "${GENERATED_DIR}" ]]; then
    rm -f "${GENERATED_DIR}"/00-configmap.yaml
    rm -f "${GENERATED_DIR}"/01-headless-service.yaml
    rm -f "${GENERATED_DIR}"/02-job-rank-*.yaml
    rm -f "${GENERATED_DIR}"/apply.sh
    log "Generated manifests removed"
  fi

  rm -f "${PROFILES_DIR}"/profile_*.json 2>/dev/null || true
  rm -f "${PLAN_FILE}" 2>/dev/null || true
  log "Profiles and HPP plan removed"

  rm -rf /tmp/baselines_tb_logs 2>/dev/null || true
  rm -rf "${BASELINES_DIR}/tb_logs" 2>/dev/null || true
  log "TensorBoard logs removed"

  rm -f /tmp/baselines_asteroid_image.tar 2>/dev/null || true
  rm -f /tmp/baselines_src.tar.gz 2>/dev/null || true
  rm -f /tmp/hf_cache.tar.gz 2>/dev/null || true
  log "Temp bundles removed"

  rm -f "${BASELINES_DIR}"/deploy_full.log 2>/dev/null || true
  log "Deploy logs removed"

  # 6. Kill local background processes
  info "Killing local background processes..."
  pkill -f "monitor_asteroid.sh" 2>/dev/null || true
  pkill -f "tensorboard.*baselines" 2>/dev/null || true
  pkill -f "kubectl.*logs.*${APP_LABEL}" 2>/dev/null || true
  log "Local processes cleaned"

  # 7. Reset GPU memory on all nodes
  info "Resetting GPU memory on all nodes..."
  if [[ -n "$node_ips" ]]; then
    while IFS= read -r ip; do
      ssh -n -o ConnectTimeout=5 -o StrictHostKeyChecking=no "ubuntu@${ip}" \
        "python3 -c 'import torch; torch.cuda.empty_cache()' 2>/dev/null || true" \
        2>/dev/null || true
    done <<< "$node_ips"
    log "GPU memory reset"
  fi

  # Summary
  echo ""
  header "Cleanup Complete"
  echo ""
  echo "  Destroyed:"
  echo "    - All K8s jobs, pods, services, configmaps"
  echo "    - All GPU processes on every cluster node"
  echo "    - NCCL/torch shared memory on every node"
  echo "    - Remote checkpoint files on every node"
  echo "    - Generated manifests, profiles, HPP plan"
  echo "    - TensorBoard logs, local background processes"
  echo "    - Temp bundles, deploy logs"
  echo ""
  echo "  Preserved:"
  echo "    - K3s cluster (nodes still Ready)"
  echo "    - Docker images (cached on nodes)"
  echo "    - NVIDIA device plugin (still running)"
  echo "    - Local checkpoint collections"
  echo "    - Source code, config YAML, deploy script"
  echo ""
  echo "  Next steps:"
  echo "    $(basename "$0") --status                    # Verify clean state"
  echo "    $(basename "$0") --skip-k3s --skip-gpu       # Full redeploy"
  echo "    $(basename "$0") --redeploy                  # Quick (skip build/profile/plan)"
}

# ============================================================================
# Status
# ============================================================================

show_status() {
  local watch_mode="${1:-false}"
  local interval="${STATUS_INTERVAL:-5}"

  _print_status() {
    if [[ "$watch_mode" == "true" ]]; then
      clear
      echo "  Baselines-Asteroid Status  (every ${interval}s — Ctrl-C to quit)"
      echo "  $(date '+%Y-%m-%d %H:%M:%S')"
      echo "──────────────────────────────────────────────────────────"
    else
      header "BASELINES-ASTEROID STATUS"
    fi
    echo ""
    echo "Nodes:"
    kubectl get nodes -o wide 2>/dev/null || echo "  kubectl not configured"
    echo ""
    echo "GPU Resources:"
    kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable."nvidia\.com/gpu" 2>/dev/null || true
    echo ""
    echo "Training Pods:"
    kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" -o wide 2>/dev/null || echo "  No training pods"
    echo ""
    echo "Training Jobs:"
    kubectl get jobs -n "${NAMESPACE}" -l "app=${APP_LABEL}" 2>/dev/null || echo "  No training jobs"
    echo ""

    # Show last training log line
    local last_pod
    last_pod=$(kubectl get pods -n "${NAMESPACE}" -l "app=${APP_LABEL}" --no-headers --sort-by=.metadata.name 2>/dev/null | tail -1 | awk '{print $1}')
    if [[ -n "$last_pod" ]]; then
      echo "Latest Training Output (${last_pod}):"
      kubectl logs -n "${NAMESPACE}" "${last_pod}" --tail=10 2>/dev/null | grep -E "iter|loss|epoch" || echo "  (no training output yet)"
    fi
  }

  if [[ "$watch_mode" == "true" ]]; then
    trap 'echo ""; echo "Status watch stopped."; exit 0' INT
    while true; do
      _print_status
      sleep "$interval"
    done
  else
    _print_status
  fi
}

# ============================================================================
# Usage
# ============================================================================

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run the full Baselines-Asteroid deployment pipeline or individual phases.

Options:
  --phase PHASE       Run a single phase:
                      k3s|registry|gpu|profile|plan|build|manifests|
                      deploy|monitor|tensorboard|mps
  --strategy NAME     Override parallelism strategy: asteroid | confident | dtfm
  --config PATH       Path to config YAML (default: configs/asteroid_default.yaml)
  --skip-k3s          Skip K3s installation (already installed)
  --skip-registry     Skip registry setup (already running)
  --skip-gpu          Skip GPU/NVIDIA setup (already configured)
  --skip-profile      Skip profiling (use existing profiles)
  --skip-plan         Skip HPP planning (use existing plan)
  --skip-build        Skip Docker image build (use existing image)
  --redeploy          Only regenerate manifests and redeploy jobs
  --monitor           Launch training monitor after deployment
  --status            Show current cluster and training status
  --status --watch    Continuously refresh status (every 5s, Ctrl-C to stop)
  --checkpoints [DIR] Collect saved checkpoints from nodes (default: ./checkpoints_collected)
  --merge-checkpoints [DIR] [ITER]  Merge rank checkpoints into single model file
  --stop              Stop all training: delete jobs, pods, services, clean up
  -h, --help          Show this help message

Phases (run in order):
  1.  k3s         Install K3s cluster
  1b. registry    Setup private Docker registry on master
  2.  gpu         Setup NVIDIA runtime & device plugin
  3.  profile     Profile cluster hardware
  4.  plan        Run HPP optimizer
  5.  build       Build & push Docker image to registry
  6.  manifests   Generate K8s job manifests
  7.  deploy      Apply manifests & start training
  8.  monitor     Live training dashboard
  9.  tensorboard Persistent TensorBoard dashboard
  10. mps         Start NVIDIA MPS on all nodes (opt-in)

Examples:
  $(basename "$0")                              # Full deployment
  $(basename "$0") --skip-k3s --skip-gpu        # Skip infra (already set up)
  $(basename "$0") --redeploy                   # Quick redeploy after code change
  $(basename "$0") --phase monitor              # Just monitor training
  $(basename "$0") --phase tensorboard          # Launch TensorBoard dashboard
  $(basename "$0") --status                     # Check cluster status
  $(basename "$0") --checkpoints                # Collect trained model checkpoints
  $(basename "$0") --merge-checkpoints ./ckpt 500  # Merge into single model
  $(basename "$0") --stop                       # Stop training and clean up everything
EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --phase)
        PHASE="$2"
        shift 2
        ;;
      --strategy)
        STRATEGY="$2"
        shift 2
        ;;
      --config)
        ASTEROID_CONFIG="$2"
        shift 2
        ;;
      --skip-k3s)
        RUN_K3S=0
        shift
        ;;
      --skip-registry)
        RUN_REGISTRY=0
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
        REDEPLOY_ONLY=1
        shift
        ;;
      --monitor)
        RUN_MONITOR=1
        shift
        ;;
      --status)
        shift
        if [[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]]; then
          show_status true
        else
          show_status false
        fi
        exit 0
        ;;
      --checkpoints)
        shift
        collect_checkpoints "${1:-./checkpoints_collected}"
        exit 0
        ;;
      --merge-checkpoints)
        shift
        merge_checkpoints "${1:-./checkpoints_collected}" "${2:-}"
        exit 0
        ;;
      --stop)
        shift
        stop_and_clean
        exit 0
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

# ============================================================================
# Phase Router
# ============================================================================

run_phase_by_name() {
  case "$1" in
    k3s)         phase_k3s ;;
    registry)    phase_registry ;;
    gpu)         phase_gpu ;;
    profile)     phase_profile ;;
    plan)        phase_plan ;;
    build)       phase_build ;;
    manifests)   phase_manifests ;;
    deploy)      phase_deploy ;;
    monitor)     phase_monitor ;;
    tensorboard) phase_tensorboard ;;
    mps)         phase_mps ;;
    *)
      err "Unknown phase: $1"
      usage
      exit 1
      ;;
  esac
}

# ============================================================================
# Main
# ============================================================================

main() {
  parse_args "$@"

  header "Baselines-Asteroid Deployment Pipeline"

  # Load YAML config (overrides image name/tag/strategy)
  load_yaml_config

  # Always regenerate inventory.ini from config to stay in sync
  if [[ -f "${ASTEROID_CONFIG}" ]]; then
    generate_inventory "${ASTEROID_CONFIG}" "${ANSIBLE_INVENTORY}"
  fi

  # Check all prerequisites
  check_prereqs

  echo "  Image:    ${IMAGE_FULL}"
  echo "  Strategy: ${STRATEGY:-auto}"
  echo "  Config:   ${ASTEROID_CONFIG}"
  echo "  Secrets:  ${ANSIBLE_SECRETS}"
  echo ""

  # Single phase mode
  if [[ -n "${PHASE}" ]]; then
    run_phase_by_name "${PHASE}"
    exit 0
  fi

  # Redeploy mode (fastest path for code changes)
  if [[ "${REDEPLOY_ONLY}" -eq 1 ]]; then
    phase_manifests
    phase_deploy
    if [[ "${RUN_MONITOR}" -eq 1 ]]; then
      phase_monitor
    fi
    exit 0
  fi

  # Full pipeline
  local start_time
  start_time=$(date +%s)

  if [[ "${RUN_K3S}" -eq 1 ]]; then
    phase_k3s
  fi

  if [[ "${RUN_REGISTRY}" -eq 1 ]]; then
    phase_registry
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

  phase_manifests
  phase_deploy

  local end_time duration_min
  end_time=$(date +%s)
  duration_min=$(( (end_time - start_time) / 60 ))

  header "BASELINES-ASTEROID DEPLOYMENT COMPLETE"
  echo ""
  echo "  Duration:    ${duration_min} minutes"
  echo "  Image:       ${IMAGE_FULL}"
  echo "  Plan:        ${PLAN_FILE}"
  echo "  Manifests:   ${GENERATED_DIR}/"
  echo ""
  echo "  Useful commands:"
  echo "    $(basename "$0") --status              # Check cluster status"
  echo "    $(basename "$0") --phase monitor       # Live training dashboard"
  echo "    $(basename "$0") --phase tensorboard   # TensorBoard"
  echo "    kubectl logs -n ${NAMESPACE} -f -l app=${APP_LABEL}  # Stream all logs"
  echo "    $(basename "$0") --stop                # Stop training & cleanup"
  echo ""

  if [[ "${RUN_MONITOR}" -eq 1 ]]; then
    phase_monitor
  fi
}

main "$@"
