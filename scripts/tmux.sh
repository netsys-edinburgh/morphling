#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="device_emulator"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n server "cd ${ROOT_DIR} && ./scripts/server.sh"
tmux new-window -t "${SESSION_NAME}" -n devices "cd ${ROOT_DIR} && sleep 20s & ./scripts/start_multiple_devices.sh 100 0"

tmux select-window -t "${SESSION_NAME}:server"
tmux attach -t "${SESSION_NAME}"
