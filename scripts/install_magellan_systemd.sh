#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:-}"
REPO_ROOT="${MAGELLAN_REPOSITORY_ROOT:-$HOME/Magellan-V2}"
SERVICE_NAME="${MAGELLAN_SYSTEMD_SERVICE:-magellan}"
RUN_USER="${MAGELLAN_RUN_USER:-${USER:-WILL}}"
SSH_USER="${MAGELLAN_SSH_USER:-$RUN_USER}"

if [[ -z "$NODE_ID" ]]; then
  echo "usage: $0 <node-id>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
if [[ -z "$GIT_BRANCH" ]]; then
  GIT_BRANCH="DETACHED"
fi
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

cat > "$TMP_FILE" <<EOF
[Unit]
Description=Magellan V2 decentralized scheduler (${NODE_ID})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_ROOT}
Environment=MAGELLAN_NODE_ID=${NODE_ID}
Environment=MAGELLAN_GIT_SHA=${GIT_SHA}
Environment=MAGELLAN_GIT_BRANCH=${GIT_BRANCH}
Environment=MAGELLAN_CONFIG=config/cluster.gcp.json
Environment=MAGELLAN_POLICY=config/policy.prod.json
Environment=MAGELLAN_DATASETS=datasets
Environment=MAGELLAN_STATE_ROOT=${REPO_ROOT}/runtime-state-gcp
Environment=MAGELLAN_REMOTE_STATE_ROOT=${REPO_ROOT}/runtime-state-gcp
Environment=MAGELLAN_REPOSITORY_ROOT=${REPO_ROOT}
Environment=MAGELLAN_SSH_USER=${SSH_USER}
Environment=MAGELLAN_TASK_FILES=
Environment=PYTHONUNBUFFERED=1
ExecStart=${REPO_ROOT}/.venv/bin/python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
KillMode=control-group

[Install]
WantedBy=multi-user.target
EOF

sudo install -m 0644 "$TMP_FILE" "$UNIT_FILE"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl --no-pager --full status "$SERVICE_NAME"

echo "SYSTEMD INSTALL PASSED node=$NODE_ID service=$SERVICE_NAME"
