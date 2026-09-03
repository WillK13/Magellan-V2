#!/usr/bin/env bash
set -euo pipefail

NODE_ID="${1:-}"
REPO_ROOT="${MAGELLAN_REPOSITORY_ROOT:-$HOME/Magellan-V2}"
SERVICE_NAME="${MAGELLAN_SYSTEMD_SERVICE:-magellan}"
RUN_USER="${MAGELLAN_RUN_USER:-${USER:-WILL}}"
SSH_USER="${MAGELLAN_SSH_USER:-$RUN_USER}"
CLEAR_DROPINS="${MAGELLAN_CLEAR_SYSTEMD_DROPINS:-0}"
PREPARE_STATE_ROOT="${MAGELLAN_PREPARE_STATE_ROOT:-0}"
INSTALL_CARBON_METRIC="${MAGELLAN_INSTALL_CARBON_METRIC:-}"

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
STATE_ROOT="${MAGELLAN_INSTALL_STATE_ROOT:-${REPO_ROOT}/runtime-state-gcp}"
REMOTE_STATE_ROOT="${MAGELLAN_INSTALL_REMOTE_STATE_ROOT:-$STATE_ROOT}"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT


if [[ -n "$INSTALL_CARBON_METRIC" && "$INSTALL_CARBON_METRIC" != "direct" && "$INSTALL_CARBON_METRIC" != "lifecycle" ]]; then
  echo "MAGELLAN_INSTALL_CARBON_METRIC must be direct or lifecycle" >&2
  exit 3
fi

if [[ "$PREPARE_STATE_ROOT" == "1" ]]; then
  CURRENT_ENVIRONMENT="$(sudo systemctl show "$SERVICE_NAME" --property=Environment --value 2>/dev/null || true)"
  CURRENT_STATE_ROOT="$(printf '%s\n' "$CURRENT_ENVIRONMENT" \
    | tr ' ' '\n' \
    | sed -n 's/^MAGELLAN_STATE_ROOT=//p' \
    | tail -1)"

  if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    ACTIVE_COUNT="$(curl -fsS --max-time 10 http://127.0.0.1:8040/tasks       | "$REPO_ROOT/.venv/bin/python" -c 'import json,sys; data=json.load(sys.stdin); active={"running","paused","migrating","recovering"}; print(sum((item.get("state") or {}).get("status") in active for item in data.get("tasks", [])))')"
    if [[ "$ACTIVE_COUNT" != "0" ]]; then
      echo "refusing state-root transition with $ACTIVE_COUNT active task(s)" >&2
      exit 14
    fi
  fi

  if [[ -n "$CURRENT_STATE_ROOT" && "$CURRENT_STATE_ROOT" != "$STATE_ROOT" ]]; then
    if [[ -e "$STATE_ROOT" ]] && [[ -n "$(find "$STATE_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      ARCHIVE_ROOT="${STATE_ROOT}.pre-stage5a1-$(date -u +%Y%m%dT%H%M%SZ)-$$"
      mv "$STATE_ROOT" "$ARCHIVE_ROOT"
      echo "Archived inactive prior state root: $STATE_ROOT -> $ARCHIVE_ROOT"
    fi
  fi
fi

mkdir -p "$STATE_ROOT"
if [[ "$REMOTE_STATE_ROOT" == "$STATE_ROOT" ]]; then
  :
elif [[ "$REMOTE_STATE_ROOT" == "$REPO_ROOT"/* ]]; then
  mkdir -p "$REMOTE_STATE_ROOT"
fi

if [[ "$CLEAR_DROPINS" == "1" ]]; then
  sudo rm -rf     "/etc/systemd/system/${SERVICE_NAME}.service.d"     "/run/systemd/system/${SERVICE_NAME}.service.d"
fi

CARBON_ENV_LINE=""
if [[ -n "$INSTALL_CARBON_METRIC" ]]; then
  CARBON_ENV_LINE="Environment=MAGELLAN_CARBON_METRIC=${INSTALL_CARBON_METRIC}"
fi

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
${CARBON_ENV_LINE}
Environment=MAGELLAN_STATE_ROOT=${STATE_ROOT}
Environment=MAGELLAN_REMOTE_STATE_ROOT=${REMOTE_STATE_ROOT}
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
