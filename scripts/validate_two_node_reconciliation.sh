#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
TASK_ID="${TASK_ID:-counter-reconciliation-001}"
STATE_ROOT_NAME="${STATE_ROOT_NAME:-runtime-state-reconciliation}"

state_json() {
  local api="$1" task="$2"
  curl -fsS "${api}/tasks" | python -c '
import json,sys
payload=json.load(sys.stdin)
for item in payload["tasks"]:
    if item["state"]["task_id"] == sys.argv[1]:
        print(json.dumps(item["state"]))
        raise SystemExit(0)
raise SystemExit(f"task not found: {sys.argv[1]}")
' "$task"
}

state_field() {
  state_json "$1" "$2" | python -c '
import json,sys
value=json.load(sys.stdin).get(sys.argv[1])
print("" if value is None else value)
' "$3"
}

wait_for_owner() {
  local api="$1" task="$2" owner="$3" timeout="$4"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    local current
    current="$(state_field "$api" "$task" owner_node_id 2>/dev/null || true)"
    echo "task=${task} owner=${current:-unknown} wanted=${owner}"
    [[ "$current" == "$owner" ]] && return 0
    sleep 1
  done
  return 1
}

cleanup() {
  curl -sS -X POST "${BOSTON_API}/tasks/${TASK_ID}/stop" >/dev/null 2>&1 || true
  curl -sS -X POST "${VIRGINIA_API}/tasks/${TASK_ID}/stop" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== Peer health and reconciliation status =="
curl -fsS "${BOSTON_API}/health" | python -m json.tool
curl -fsS "${VIRGINIA_API}/health" | python -m json.tool
curl -fsS "${BOSTON_API}/ownership/snapshot" >/tmp/boston-ownership-snapshot.json
curl -fsS "${VIRGINIA_API}/ownership/snapshot" >/tmp/virginia-ownership-snapshot.json
python -m json.tool /tmp/boston-ownership-snapshot.json
python -m json.tool /tmp/virginia-ownership-snapshot.json

echo "== Start and migrate with a deliberately lost activation response =="
curl -fsS -X POST "${BOSTON_API}/tasks/${TASK_ID}/start" | python -m json.tool
sleep 3
curl -fsS -X POST \
  "${BOSTON_API}/tasks/${TASK_ID}/migrate/virginia" \
  | tee /tmp/reconciliation-migration.json \
  | python -m json.tool

python - <<'PY'
import json
payload=json.load(open('/tmp/reconciliation-migration.json'))
assert payload.get('migrated') is True, payload
assert payload['state']['owner_node_id'] == 'virginia', payload
print('source resolved the destination activation as committed')
PY

wait_for_owner "$BOSTON_API" "$TASK_ID" virginia 20
wait_for_owner "$VIRGINIA_API" "$TASK_ID" virginia 20

curl -fsS "${BOSTON_API}/migrations" >/tmp/boston-migrations.json
read -r bid_id migration_id < <(python - "$TASK_ID" <<'PY'
import json,sys
records=json.load(open('/tmp/boston-migrations.json'))
matching=[r for r in records if r['task_id'] == sys.argv[1]]
if not matching:
    raise SystemExit('migration record not found')
record=matching[-1]
print(record['bid_id'], record['migration_id'])
PY
)

echo "migration_id=${migration_id}"
curl -fsS "${BOSTON_API}/migrations/${migration_id}" \
  | tee /tmp/source-migration-record.json | python -m json.tool
curl -fsS "${VIRGINIA_API}/migrations/${migration_id}" \
  | tee /tmp/destination-migration-record.json | python -m json.tool

python - <<'PY'
import json
source=json.load(open('/tmp/source-migration-record.json'))
destination=json.load(open('/tmp/destination-migration-record.json'))
assert source['role'] == 'source', source
assert destination['role'] == 'destination', destination
assert source['status'] == 'activated', source
assert destination['status'] == 'activated', destination
print('both durable migration journals agree on ACTIVATED')
PY

curl -fsS "${VIRGINIA_API}/bids/${bid_id}" \
  | tee /tmp/durable-bid.json | python -m json.tool
python - <<'PY'
import json
bid=json.load(open('/tmp/durable-bid.json'))
assert bid['status'] == 'consumed', bid
print('destination reservation is durably consumed')
PY

echo "== Durable files exist on Virginia =="
ssh "$VIRGINIA_SSH" "
  test -s ~/Magellan-V2/${STATE_ROOT_NAME}/control/bids.json &&
  test -s ~/Magellan-V2/${STATE_ROOT_NAME}/control/migrations/${migration_id}.json &&
  echo DURABLE_CONTROL_FILES_OK
"

trap - EXIT
cleanup

echo "ALL TWO-NODE DURABLE RECONCILIATION CHECKS PASSED"
