#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
DEFINITION_FILE="${DEFINITION_FILE:-config/submissions/dev-counter-definition.json}"
RUN_FILE="${RUN_FILE:-config/submissions/dev-counter-run.json}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

wait_run_field() {
  local api="$1" run_id="$2" expression="$3" wanted="$4"
  for attempt in $(seq 1 60); do
    response="$(curl -fsS "$api/task-runs/$run_id" 2>/dev/null || true)"
    if [[ -n "$response" ]]; then
      actual="$(printf '%s' "$response" | json_field "$expression")"
      echo "run=$run_id field=$expression actual=$actual wanted=$wanted"
      if [[ "$actual" == "$wanted" ]]; then
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for $run_id $expression=$wanted" >&2
  return 1
}

echo "== Peer health =="
curl -fsS "$BOSTON_API/health" | python -m json.tool
curl -fsS "$VIRGINIA_API/health" | python -m json.tool

echo "== Submit immutable definition to Boston =="
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary "@$DEFINITION_FILE" \
  "$BOSTON_API/task-definitions" \
  | tee /tmp/magellan-dynamic-definition.json \
  | python -m json.tool

DEFINITION_REVISION="$(cat /tmp/magellan-dynamic-definition.json | json_field 'value["revision"]')"
DEFINITION_DIGEST="$(cat /tmp/magellan-dynamic-definition.json | json_field 'value["digest"]')"
[[ "$DEFINITION_REVISION" == "1" ]]
[[ ${#DEFINITION_DIGEST} -eq 64 ]]

# Same payload is idempotent and returns the same immutable revision.
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary "@$DEFINITION_FILE" \
  "$BOSTON_API/task-definitions" \
  > /tmp/magellan-dynamic-definition-retry.json
python - <<'PY'
import json
first=json.load(open('/tmp/magellan-dynamic-definition.json'))
second=json.load(open('/tmp/magellan-dynamic-definition-retry.json'))
assert first['revision'] == second['revision']
assert first['digest'] == second['digest']
print('definition retry was idempotent')
PY

echo "== Create and auto-start a task run =="
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary "@$RUN_FILE" \
  "$BOSTON_API/task-runs" \
  | tee /tmp/magellan-dynamic-run.json \
  | python -m json.tool

RUN_ID="$(cat /tmp/magellan-dynamic-run.json | json_field 'value["run"]["run_id"]')"
[[ "$RUN_ID" == run-* ]]
wait_run_field "$BOSTON_API" "$RUN_ID" 'value["state"]["status"]' running

# Same idempotency key must return the same task run.
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary "@$RUN_FILE" \
  "$BOSTON_API/task-runs" \
  > /tmp/magellan-dynamic-run-retry.json
python - <<'PY'
import json
first=json.load(open('/tmp/magellan-dynamic-run.json'))
second=json.load(open('/tmp/magellan-dynamic-run-retry.json'))
assert first['run']['run_id'] == second['run']['run_id']
print('task-run retry was idempotent:', first['run']['run_id'])
PY

echo "== Catalog anti-entropy installs the task on Virginia =="
wait_run_field "$VIRGINIA_API" "$RUN_ID" 'value["state"]["owner_node_id"]' boston
wait_run_field "$VIRGINIA_API" "$RUN_ID" 'value["state"]["status"]' remote

curl -fsS "$VIRGINIA_API/task-definitions/dynamic-counter" \
  | tee /tmp/magellan-virginia-definition.json \
  | python -m json.tool
python - <<'PY'
import json
source=json.load(open('/tmp/magellan-dynamic-definition.json'))
remote=json.load(open('/tmp/magellan-virginia-definition.json'))
assert source['digest'] == remote['digest']
assert source['revision'] == remote['revision']
print('definition digest replicated to Virginia')
PY

echo "== Pause and resume the dynamically submitted task =="
curl -fsS -X POST "$BOSTON_API/tasks/$RUN_ID/pause?idle_seconds=120" \
  | python -m json.tool
wait_run_field "$BOSTON_API" "$RUN_ID" 'value["state"]["status"]' paused
wait_run_field "$BOSTON_API" "$RUN_ID" 'value["state"]["status"]' running

echo "== Dynamic task bids for Virginia capacity and migrates =="
curl -fsS -X POST "$BOSTON_API/tasks/$RUN_ID/migrate/virginia" \
  | tee /tmp/magellan-dynamic-migration.json \
  | python -m json.tool
python - <<'PY'
import json
payload=json.load(open('/tmp/magellan-dynamic-migration.json'))
assert payload['migrated'] is True, payload
print('migration accepted')
PY

wait_run_field "$BOSTON_API" "$RUN_ID" 'value["state"]["owner_node_id"]' virginia
for attempt in $(seq 1 60); do
  STATUS="$(curl -fsS "$VIRGINIA_API/task-runs/$RUN_ID" | json_field 'value["state"]["status"]')"
  echo "Virginia run status=$STATUS"
  if [[ "$STATUS" == "running" || "$STATUS" == "completed" ]]; then
    break
  fi
  sleep 1
done

echo "== Verify task-bid direction and structured task context =="
curl -fsS "$VIRGINIA_API/bids" > /tmp/magellan-dynamic-bids.json
python - "$RUN_ID" <<'PY'
import json,sys
run_id=sys.argv[1]
bids=json.load(open('/tmp/magellan-dynamic-bids.json'))
matching=[bid for bid in bids if bid['task_id']==run_id]
assert matching, bids
bid=matching[-1]
assert bid['bidder_type']=='task', bid
assert bid['destination_node_id']=='virginia', bid
assert bid['task_context']['priority']==10, bid
assert bid['task_context']['resource_request']['cpu_cores']==1, bid
assert bid['status']=='consumed', bid
print('task bid competed for Virginia capacity with structured task context')
PY

echo "== Wait for completion and final output on Virginia =="
wait_run_field "$VIRGINIA_API" "$RUN_ID" 'value["state"]["status"]' completed
curl -fsS "$VIRGINIA_API/tasks/$RUN_ID/outputs" \
  | tee /tmp/magellan-dynamic-output-manifest.json \
  | python -m json.tool
curl -fsS "$VIRGINIA_API/tasks/$RUN_ID/outputs/result.json" \
  | tee /tmp/magellan-dynamic-result.json \
  | python -m json.tool
python - "$RUN_ID" <<'PY'
import json,sys
run_id=sys.argv[1]
result=json.load(open('/tmp/magellan-dynamic-result.json'))
assert result['task_id']==run_id
assert result['final_value']==120
assert result['node_id']=='virginia'
print('dynamic run completed on Virginia')
PY

echo "== Create a second run from the same definition =="
python - <<'PY' > /tmp/magellan-second-run.json
import json,uuid
payload=json.load(open('config/submissions/dev-counter-run.json'))
payload['idempotency_key']='dynamic-second-'+str(uuid.uuid4())
payload['auto_start']=False
print(json.dumps(payload))
PY
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/magellan-second-run.json \
  "$BOSTON_API/task-runs" \
  | tee /tmp/magellan-second-run-response.json \
  | python -m json.tool
SECOND_RUN_ID="$(cat /tmp/magellan-second-run-response.json | json_field 'value["run"]["run_id"]')"
[[ "$SECOND_RUN_ID" != "$RUN_ID" ]]
[[ "$(cat /tmp/magellan-second-run-response.json | json_field 'value["state"]["status"]')" == "stopped" ]]

echo "== Durable catalog files exist on both peers =="
test -s runtime-state-dynamic-submission/control/task_catalog.json
ssh "$VIRGINIA_SSH" \
  'test -s ~/Magellan-V2/runtime-state-dynamic-submission/control/task_catalog.json && echo VIRGINIA_CATALOG_OK'

echo "ALL TWO-NODE DYNAMIC TASK SUBMISSION CHECKS PASSED"
