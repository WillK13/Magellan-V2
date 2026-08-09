#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:-${MAGELLAN_NODE_ID:-}}"
REPO_ROOT="${MAGELLAN_REPOSITORY_ROOT:-$HOME/Magellan-V2}"
STATE_ROOT_NAME="${MAGELLAN_STATE_ROOT_NAME:-runtime-state-gcp}"

if [[ -z "$NODE_ID" ]]; then
  echo "usage: $0 <node-id>" >&2
  exit 2
fi

cd "$REPO_ROOT"
source .venv/bin/activate

unset MAGELLAN_TEST_FORCE_ACTIVATION_RESPONSE_LOSS

export MAGELLAN_NODE_ID="$NODE_ID"
export MAGELLAN_CONFIG="${MAGELLAN_CONFIG:-config/cluster.gcp.json}"
export MAGELLAN_POLICY="${MAGELLAN_POLICY:-config/policy.prod.json}"
export MAGELLAN_DATASETS="${MAGELLAN_DATASETS:-datasets}"
export MAGELLAN_STATE_ROOT="${MAGELLAN_STATE_ROOT:-$REPO_ROOT/$STATE_ROOT_NAME}"
export MAGELLAN_REMOTE_STATE_ROOT="${MAGELLAN_REMOTE_STATE_ROOT:-$REPO_ROOT/$STATE_ROOT_NAME}"
export MAGELLAN_REPOSITORY_ROOT="$REPO_ROOT"
export MAGELLAN_SSH_USER="${MAGELLAN_SSH_USER:-${USER:-WILL}}"
export MAGELLAN_TASK_FILES="${MAGELLAN_TASK_FILES:-}"
export PYTHONUNBUFFERED=1

exec python -m uvicorn magellan.api.app:app \
  --host 0.0.0.0 \
  --port 8040
