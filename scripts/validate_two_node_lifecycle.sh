#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://10.142.0.2:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
BOSTON_SSH="${BOSTON_SSH:-WILL@10.142.0.2}"
TASK_ID="${TASK_ID:-counter-complete-001}"
RECOVERY_TASK_ID="${RECOVERY_TASK_ID:-counter-recovery-001}"
EXPECTED_FINAL_VALUE="${EXPECTED_FINAL_VALUE:-80}"

state_field() {
  local api="$1" task="$2" field="$3"
  curl -fsS "${api}/tasks" | python -c '
import json,sys
payload=json.load(sys.stdin)
task_id=sys.argv[1]
field=sys.argv[2]
for item in payload["tasks"]:
    if item["state"]["task_id"] == task_id:
        value=item["state"].get(field)
        print("" if value is None else value)
        raise SystemExit(0)
raise SystemExit(f"task not found: {task_id}")
' "$task" "$field"
}

wait_for_state() {
  local api="$1" task="$2" wanted="$3" timeout="$4"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    local current
    current="$(state_field "$api" "$task" status || true)"
    if [[ "$current" == "$wanted" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${task}=${wanted} on ${api}" >&2
  return 1
}

echo "== Peer health =="
curl -fsS "${BOSTON_API}/health" | python -m json.tool
curl -fsS "${VIRGINIA_API}/health" | python -m json.tool

echo "== Start finite task in Boston =="
curl -fsS -X POST "${BOSTON_API}/tasks/${TASK_ID}/start" | python -m json.tool
sleep 2

echo "== Force an operator migration through scoring + bidding + reservation =="
curl -fsS -X POST \
  "${BOSTON_API}/tasks/${TASK_ID}/migrate/virginia" \
  | python -m json.tool

echo "== Wait for completion on Virginia =="
wait_for_state "$VIRGINIA_API" "$TASK_ID" completed 120
owner="$(state_field "$VIRGINIA_API" "$TASK_ID" owner_node_id)"
[[ "$owner" == "virginia" ]] || {
  echo "Expected owner=virginia, got ${owner}" >&2
  exit 1
}

curl -fsS "${VIRGINIA_API}/tasks/${TASK_ID}/outputs" \
  | tee /tmp/magellan-final-manifest.json \
  | python -m json.tool
curl -fsS \
  "${VIRGINIA_API}/tasks/${TASK_ID}/outputs/result.json" \
  | tee /tmp/magellan-result.json \
  | python -m json.tool
python - "$EXPECTED_FINAL_VALUE" <<'PY'
import json,sys
expected=int(sys.argv[1])
with open('/tmp/magellan-result.json') as f:
    result=json.load(f)
assert result['final_value'] == expected, result
assert result['node_id'] == 'virginia', result
print('final output validated')
PY

echo "== Validate crash recovery on Boston =="
curl -fsS -X POST \
  "${BOSTON_API}/tasks/${RECOVERY_TASK_ID}/start" \
  | python -m json.tool
old_pid="$(state_field "$BOSTON_API" "$RECOVERY_TASK_ID" pid)"
ssh -o BatchMode=yes "$BOSTON_SSH" "kill -9 ${old_pid}"

deadline=$((SECONDS + 45))
while (( SECONDS < deadline )); do
  status="$(state_field "$BOSTON_API" "$RECOVERY_TASK_ID" status || true)"
  new_pid="$(state_field "$BOSTON_API" "$RECOVERY_TASK_ID" pid || true)"
  attempts="$(state_field "$BOSTON_API" "$RECOVERY_TASK_ID" recovery_attempts || true)"
  if [[ "$status" == "running" && -n "$new_pid" && "$new_pid" != "$old_pid" && "${attempts:-0}" -ge 1 ]]; then
    echo "recovery validated: old_pid=${old_pid} new_pid=${new_pid} attempts=${attempts}"
    break
  fi
  sleep 1
done
[[ "$(state_field "$BOSTON_API" "$RECOVERY_TASK_ID" status)" == "running" ]]

curl -fsS -X POST \
  "${BOSTON_API}/tasks/${RECOVERY_TASK_ID}/stop" \
  >/dev/null

echo "== Validate accepted reservation expiry on Virginia =="
probe_json="$(python - <<'PY'
import json,uuid
from datetime import datetime,timezone
bid_id='expiry-'+str(uuid.uuid4())
print(json.dumps({
  'bid_id': bid_id,
  'epoch_id': 'expiry-probe',
  'task_id': 'reservation-expiry-probe',
  'source_node_id': 'boston',
  'destination_node_id': 'virginia',
  'candidate': {
    'action':'migrate',
    'source_node_id':'boston',
    'destination_node_id':'virginia',
    'time_seconds':1,
    'carbon_grams':1,
    'cost_usd':0,
    'details':{},
    'normalized_time':0,
    'normalized_carbon':0,
    'normalized_cost':0,
    'score':0,
  },
  'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
}))
PY
)"
bid_id="$(python -c 'import json,sys; print(json.load(sys.stdin)["bid_id"])' <<<"$probe_json")"
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  -d "$probe_json" \
  "${VIRGINIA_API}/bids" >/dev/null
sleep 5
status="$(curl -fsS "${VIRGINIA_API}/bids/${bid_id}" | python -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
[[ "$status" == "accepted" ]] || {
  echo "Expected accepted reservation, got ${status}" >&2
  exit 1
}
sleep 16
status="$(curl -fsS "${VIRGINIA_API}/bids/${bid_id}" | python -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
[[ "$status" == "expired" ]] || {
  echo "Expected expired reservation, got ${status}" >&2
  exit 1
}
echo "reservation expiry validated"

echo "ALL TWO-NODE LIFECYCLE CHECKS PASSED"
