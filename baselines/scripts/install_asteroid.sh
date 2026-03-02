#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"

VENV_DIR="${BASELINES_DIR}/.venv_asteroid"
DEPLOY_DIR="${BASELINES_DIR}/deploy_asteroid"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FORCE=0
SKIP_TORCH=0
SKIP_CUPY=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[install-asteroid]${NC} $*"; }
info() { echo -e "${BLUE}[install-asteroid]${NC} $*"; }
warn() { echo -e "${YELLOW}[install-asteroid]${NC} $*"; }
err() { echo -e "${RED}[install-asteroid]${NC} $*" >&2; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --venv-dir <path>     Virtualenv directory (default: baselines/.venv_asteroid)
  --deploy-dir <path>   Deployment template directory (default: baselines/deploy_asteroid)
  --python <path>       Python executable to bootstrap venv (default: python3)
  --skip-torch          Do not install/upgrade torch packages
  --skip-cupy           Do not install CuPy
  --force               Overwrite existing template files
  -h, --help            Show this help

Examples:
  bash baselines/scripts/install_asteroid.sh
  bash baselines/scripts/install_asteroid.sh --venv-dir /opt/asteroid-venv --force
  bash baselines/scripts/install_asteroid.sh --skip-cupy
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --skip-cupy)
      SKIP_CUPY=1
      shift
      ;;
    --force)
      FORCE=1
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

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    err "Required command not found: $cmd"
    exit 1
  fi
}

detect_arch() {
  uname -m
}

detect_cuda_version() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
  fi

  if [[ -z "${ver}" ]] && command -v nvcc >/dev/null 2>&1; then
    ver="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n1 || true)"
  fi

  if [[ -z "${ver}" ]]; then
    echo "0.0"
  else
    echo "${ver}"
  fi
}

detect_jetpack_major() {
  if [[ -f /etc/nv_tegra_release ]]; then
    sed -n 's/.* R\([0-9][0-9]*\).*/\1/p' /etc/nv_tegra_release | head -n1
  else
    echo "0"
  fi
}

install_system_deps_if_possible() {
  local missing=()
  command -v sshpass >/dev/null 2>&1 || missing+=(sshpass)
  command -v iperf3 >/dev/null 2>&1 || missing+=(iperf3)
  command -v curl >/dev/null 2>&1 || missing+=(curl)

  if [[ ${#missing[@]} -eq 0 ]]; then
    info "System deployment tools already present (sshpass/iperf3/curl)."
    return
  fi

  warn "Missing system tools: ${missing[*]}"
  if command -v apt-get >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
    info "Installing missing system tools with apt-get..."
    sudo apt-get update -y
    sudo apt-get install -y "${missing[@]}"
  else
    warn "Could not auto-install system tools (need apt-get + sudo)."
    warn "Install manually: ${missing[*]}"
  fi
}

create_venv() {
  require_cmd "${PYTHON_BIN}"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    info "Creating virtualenv at ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  else
    info "Using existing virtualenv at ${VENV_DIR}"
  fi
}

pip_install_common() {
  local pip_bin="${VENV_DIR}/bin/pip"
  info "Upgrading pip/setuptools/wheel (setuptools<81 for TensorBoard compatibility)"
  "${pip_bin}" install --upgrade pip "setuptools<81" wheel

  info "Installing Asteroid deployment Python dependencies"
  "${pip_bin}" install \
    "numpy>=1.24,<2.0" \
    "pyyaml>=6,<7" \
    "typing_extensions>=4.8" \
    "scipy>=1.10" \
    "transformers>=4.40,<6" \
    "pydantic>=2,<3" \
    "pydantic-settings>=2,<3" \
    "jinja2>=3.1,<4" \
    "tensorboard>=2.10,<2.21" \
    "datasets>=2.14,<5" \
    "ansible>=9,<11"
}

install_torch_for_platform() {
  local arch="$1"
  local cuda_ver="$2"
  local jetpack_major="$3"
  local pip_bin="${VENV_DIR}/bin/pip"

  if [[ "${SKIP_TORCH}" -eq 1 ]]; then
    warn "Skipping torch installation (--skip-torch)."
    return
  fi

  local cuda_major="${cuda_ver%%.*}"
  local cuda_minor="${cuda_ver#*.}"

  if [[ "${arch}" == "x86_64" ]]; then
    local index_url="https://download.pytorch.org/whl/cpu"
    if [[ "${cuda_major}" -ge 12 ]]; then
      if [[ "${cuda_minor}" -ge 4 ]]; then
        index_url="https://download.pytorch.org/whl/cu124"
      else
        index_url="https://download.pytorch.org/whl/cu121"
      fi
    elif [[ "${cuda_major}" -eq 11 ]]; then
      index_url="https://download.pytorch.org/whl/cu118"
    fi

    info "Installing torch stack from ${index_url}"
    "${pip_bin}" install --index-url "${index_url}" torch torchvision torchaudio
    return
  fi

  if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
    local wheel_url=""
    if [[ "${jetpack_major}" -ge 36 ]]; then
      wheel_url="https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"
    elif [[ "${jetpack_major}" -ge 35 ]]; then
      wheel_url="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
    else
      wheel_url="https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    fi

    info "Installing Jetson torch wheel: ${wheel_url}"
    if ! "${pip_bin}" install "${wheel_url}"; then
      err "Jetson wheel install failed. Please verify Python ABI and JetPack version."
      exit 1
    fi

    if ! "${pip_bin}" install torchvision torchaudio; then
      warn "Could not install torchvision/torchaudio on ARM64. Continuing."
    fi
    return
  fi

  warn "Unknown architecture (${arch}). Installing CPU torch wheels."
  "${pip_bin}" install --index-url "https://download.pytorch.org/whl/cpu" torch torchvision torchaudio
}

install_cupy_for_cuda() {
  local arch="$1"
  local cuda_ver="$2"
  local pip_bin="${VENV_DIR}/bin/pip"

  if [[ "${SKIP_CUPY}" -eq 1 ]]; then
    warn "Skipping CuPy installation (--skip-cupy)."
    return
  fi

  local cuda_major="${cuda_ver%%.*}"
  local cupy_pkg=""

  if [[ "${cuda_major}" -ge 12 ]]; then
    cupy_pkg="cupy-cuda12x"
  elif [[ "${cuda_major}" -eq 11 ]]; then
    cupy_pkg="cupy-cuda11x"
  else
    warn "No CUDA runtime detected; skipping CuPy."
    return
  fi

  info "Installing ${cupy_pkg}"
  if ! "${pip_bin}" install "${cupy_pkg}"; then
    if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
      warn "CuPy wheel unavailable on ARM64 for this CUDA/Python combo."
      warn "Continuing with torch.distributed fallback path."
    else
      err "CuPy install failed on x86_64 (${cupy_pkg})."
      exit 1
    fi
  fi
}

write_template_file() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"

  if [[ -f "${path}" && "${FORCE}" -ne 1 ]]; then
    info "Keeping existing template: ${path}"
    cat >/dev/null
    return
  fi

  cat >"${path}"
  log "Wrote template: ${path}"
}

setup_cluster_templates() {
  info "Creating Asteroid deployment template tree at ${DEPLOY_DIR}"
  mkdir -p "${DEPLOY_DIR}/vars" "${DEPLOY_DIR}/templates" "${DEPLOY_DIR}/generated"

  write_template_file "${DEPLOY_DIR}/ansible.cfg" <<'EOF'
[defaults]
inventory = inventory.ini
vault_password_file = ~/.asteroid_vault_pass
host_key_checking = False
timeout = 30
gathering = smart
retry_files_enabled = False
stdout_callback = ansible.builtin.default
result_format = yaml

[privilege_escalation]
become = True
become_method = sudo

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s -o StrictHostKeyChecking=no
EOF

  write_template_file "${DEPLOY_DIR}/inventory.ini.template" <<'EOF'
# Asteroid deployment inventory template

[master]
# Replace ansible_host with your master node IP
master_node ansible_host=10.0.0.10 ansible_user=ubuntu rank=0 gpu_id=0 nic=eth0 hostname=master-node

[workers]
# Add one line per worker node
worker1 ansible_host=10.0.0.11 ansible_user=ubuntu rank=1 gpu_id=0 nic=eth0 hostname=worker1
worker2 ansible_host=10.0.0.12 ansible_user=ubuntu rank=2 gpu_id=0 nic=eth0 hostname=worker2

[cluster:children]
master
workers

[cluster:vars]
ansible_python_interpreter=/usr/bin/python3
asteroid_venv=/opt/asteroid/venv
asteroid_src=/opt/asteroid/src
EOF

  write_template_file "${DEPLOY_DIR}/secrets.yml.template" <<'EOF'
---
# Encrypt this file before use:
#   cp secrets.yml.template secrets.yml
#   ansible-vault encrypt secrets.yml
ansible_ssh_pass: "CHANGE_ME"
ansible_become_pass: "CHANGE_ME"
EOF

  write_template_file "${DEPLOY_DIR}/cluster.conf.template" <<'EOF'
# IP_ADDRESS   NIC   RANK   GPU_ID
10.0.0.10      eth0  0      0
10.0.0.11      eth0  1      0
10.0.0.12      eth0  2      0
EOF

  write_template_file "${DEPLOY_DIR}/vars/pytorch_sources.yml" <<'EOF'
---
# x86_64 torch index URLs
pytorch_cu124_index: "https://download.pytorch.org/whl/cu124"
pytorch_cu121_index: "https://download.pytorch.org/whl/cu121"
pytorch_cu118_index: "https://download.pytorch.org/whl/cu118"
pytorch_cpu_index: "https://download.pytorch.org/whl/cpu"

# ARM64 Jetson wheel URLs
pytorch_l4t_jp46_url: "https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
pytorch_l4t_jp5_url: "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
pytorch_l4t_jp6_url: "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.3.0-cp310-cp310-linux_aarch64.whl"

cupy_cuda12: "cupy-cuda12x"
cupy_cuda11: "cupy-cuda11x"
EOF
}

validate_installation() {
  local py_bin="${VENV_DIR}/bin/python"
  info "Running installation validation checks"

  "${py_bin}" - <<'PY'
import importlib

mods = [
    "yaml",
    "jinja2",
    "pydantic",
    "pydantic_settings",
    "tensorboard",
    "datasets",
    "ansible",
]

missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))

if missing:
    for mod, reason in missing:
        print(f"[FAIL] {mod}: {reason}")
    raise SystemExit(1)

print("[OK] Core Python dependencies import successfully")
PY

  "${py_bin}" - <<'PY'
import torch
print(f"[OK] torch={torch.__version__}")
print(f"[OK] cuda_available={torch.cuda.is_available()}")
PY

  if [[ "${SKIP_CUPY}" -ne 1 ]]; then
    if "${py_bin}" -c "import cupy, cupy.cuda.nccl; print('[OK] cupy=' + cupy.__version__)" >/dev/null 2>&1; then
      log "CuPy import validation passed"
    else
      warn "CuPy import validation failed (fallback path still usable)."
    fi
  fi

  "${VENV_DIR}/bin/ansible-playbook" --version >/dev/null
  log "Ansible CLI validation passed"

  local required_templates=(
    "${DEPLOY_DIR}/ansible.cfg"
    "${DEPLOY_DIR}/inventory.ini.template"
    "${DEPLOY_DIR}/secrets.yml.template"
    "${DEPLOY_DIR}/cluster.conf.template"
    "${DEPLOY_DIR}/vars/pytorch_sources.yml"
  )

  for f in "${required_templates[@]}"; do
    if [[ ! -f "${f}" ]]; then
      err "Missing generated template: ${f}"
      exit 1
    fi
  done
  log "Template generation validation passed"
}

main() {
  require_cmd "sed"
  require_cmd "uname"
  require_cmd "head"

  local arch
  arch="$(detect_arch)"
  local cuda_ver
  cuda_ver="$(detect_cuda_version)"
  local jetpack_major
  jetpack_major="$(detect_jetpack_major)"

  info "Repository root: ${REPO_ROOT}"
  info "Baselines dir: ${BASELINES_DIR}"
  info "Venv dir: ${VENV_DIR}"
  info "Deploy templates dir: ${DEPLOY_DIR}"
  info "Detected arch=${arch}, cuda=${cuda_ver}, jetpack_major=${jetpack_major}"

  install_system_deps_if_possible
  create_venv
  pip_install_common
  install_torch_for_platform "${arch}" "${cuda_ver}" "${jetpack_major}"
  install_cupy_for_cuda "${arch}" "${cuda_ver}"
  setup_cluster_templates
  validate_installation

  echo
  log "Asteroid installation completed successfully"
  echo "Next steps:"
  echo "  1) source \"${VENV_DIR}/bin/activate\""
  echo "  2) cp \"${DEPLOY_DIR}/inventory.ini.template\" \"${DEPLOY_DIR}/inventory.ini\""
  echo "  3) cp \"${DEPLOY_DIR}/secrets.yml.template\" \"${DEPLOY_DIR}/secrets.yml\" && ansible-vault encrypt \"${DEPLOY_DIR}/secrets.yml\""
  echo "  4) cp \"${DEPLOY_DIR}/cluster.conf.template\" \"${REPO_ROOT}/cluster.conf\""
}

main "$@"
