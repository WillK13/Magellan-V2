#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"

PAUSE_TASK_ID="${PAUSE_TASK_ID:-counter-pause-001}"
BUDGET_TASK_ID="${BUDGET_TASK_ID:-counter-budget-001}"
MIGRATION_TASK_ID="${MIGRATION_TASK_ID:-counter-accounting-migrate-001}"
PAUSE_EVALUATION_SECONDS="${PAUSE_EVALUATION_SECONDS:-300}"

state_json() {
  local api="$1" task="$2"
  curl -fsS "${api}/tasks" | python -c '
import json,sys
payload=json.load(sys.stdin)
task_id=sys.argv[1]
for item in payload["tasks"]:
    state=item["state"]
    if state["task_id"] == task_id:
        print(json.dumps(state))
        raise SystemExit(0)
raise SystemExit(f"task not found: {task_id}")
' "$task"
}

state_field() {
  local api="$1" task="$2" field="$3"
  state_json "$api" "$task" | python -c '
import json,sys
state=json.load(sys.stdin)
value=state.get(sys.argv[1])
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("")
else:
    print(value)
' "$field"
}

wait_for_state() {
  local api="$1" task="$2" wanted="$3" timeout="$4"
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    local current
    current="$(state_field "$api" "$task" status 2>/dev/null || true)"
    echo "task=${task} state=${current:-unknown} wanted=${wanted}"
    if [[ "$current" == "$wanted" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${task}=${wanted} on ${api}" >&2
  return 1
}

checkpoint_value() {
  local task="$1"
  python - "$task" <<'PY'
import json,sys
from pathlib import Path
path=Path("runtime-state-pause-accounting/tasks") / sys.argv[1] / "checkpoint/counter.json"
print(json.loads(path.read_text())["value"])
PY
}

assert_float_gt() {
  local actual="$1" threshold="$2" label="$3"
  python - "$actual" "$threshold" "$label" <<'PY'
import sys
actual=float(sys.argv[1])
threshold=float(sys.argv[2])
label=sys.argv[3]
assert actual > threshold, f"{label}: expected {actual} > {threshold}"
print(f"{label}: {actual:.9f} > {threshold:.9f}")
PY
}

cleanup() {
  curl -sS -X POST "${BOSTON_API}/tasks/${PAUSE_TASK_ID}/stop" >/dev/null 2>&1 || true
  curl -sS -X POST "${BOSTON_API}/tasks/${BUDGET_TASK_ID}/stop" >/dev/null 2>&1 || true
  curl -sS -X POST "${BOSTON_API}/tasks/${MIGRATION_TASK_ID}/stop" >/dev/null 2>&1 || true
  curl -sS -X POST "${VIRGINIA_API}/tasks/${MIGRATION_TASK_ID}/stop" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "== Peer health =="
curl -fsS "${BOSTON_API}/health" | python -m json.tool
curl -fsS "${VIRGINIA_API}/health" | python -m json.tool

echo "== Real pause/resume with one preserved PID =="
curl -fsS -X POST "${BOSTON_API}/tasks/${PAUSE_TASK_ID}/start" | python -m json.tool
wait_for_state "$BOSTON_API" "$PAUSE_TASK_ID" running 10
sleep 3

pid_before="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" pid)"
value_before_pause="$(checkpoint_value "$PAUSE_TASK_ID")"

curl -fsS -X POST \
  "${BOSTON_API}/tasks/${PAUSE_TASK_ID}/pause?idle_seconds=${PAUSE_EVALUATION_SECONDS}" \
  | python -m json.tool
wait_for_state "$BOSTON_API" "$PAUSE_TASK_ID" paused 10

pid_paused="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" pid)"
resume_wall="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" resume_wall_at_utc)"
[[ "$pid_paused" == "$pid_before" ]] || {
  echo "Pause changed PID: before=${pid_before} paused=${pid_paused}" >&2
  exit 1
}
[[ -n "$resume_wall" ]] || {
  echo "Paused state did not persist resume_wall_at_utc" >&2
  exit 1
}

value_paused_1="$(checkpoint_value "$PAUSE_TASK_ID")"
sleep 2
value_paused_2="$(checkpoint_value "$PAUSE_TASK_ID")"
[[ "$value_paused_2" == "$value_paused_1" ]] || {
  echo "Checkpoint advanced while paused: ${value_paused_1} -> ${value_paused_2}" >&2
  exit 1
}

echo "pause froze progress at value=${value_paused_2}, pid=${pid_paused}"
wait_for_state "$BOSTON_API" "$PAUSE_TASK_ID" running 20

pid_after="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" pid)"
[[ "$pid_after" == "$pid_before" ]] || {
  echo "Resume changed PID: before=${pid_before} after=${pid_after}" >&2
  exit 1
}

sleep 2
value_after_resume="$(checkpoint_value "$PAUSE_TASK_ID")"
python - "$value_after_resume" "$value_paused_2" <<'PY'
import sys
after=int(sys.argv[1]); paused=int(sys.argv[2])
assert after > paused, f"progress did not resume: paused={paused} after={after}"
print(f"progress resumed: {paused} -> {after}")
PY

runtime_seconds="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" accumulated_runtime_seconds)"
paused_seconds="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" accumulated_paused_seconds)"
compute_cost="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" accumulated_compute_cost_usd)"
compute_carbon="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" accumulated_compute_carbon_grams)"
progress_fraction="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" progress_fraction)"
progress_rate="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" progress_rate_units_per_second)"
remaining="$(state_field "$BOSTON_API" "$PAUSE_TASK_ID" estimated_remaining_seconds)"

assert_float_gt "$runtime_seconds" 0 "runtime accounting"
assert_float_gt "$paused_seconds" 0 "pause accounting"
assert_float_gt "$compute_cost" 0 "compute cost accounting"
assert_float_gt "$compute_carbon" 0 "compute carbon accounting"
assert_float_gt "$progress_fraction" 0 "progress fraction"
assert_float_gt "$progress_rate" 0 "progress rate"
assert_float_gt "$remaining" 0 "remaining-time estimate"

curl -fsS -X POST "${BOSTON_API}/tasks/${PAUSE_TASK_ID}/stop" | python -m json.tool

echo "== Live accumulated cost prunes migration at the task budget cap =="
curl -fsS -X POST "${BOSTON_API}/tasks/${BUDGET_TASK_ID}/start" | python -m json.tool
wait_for_state "$BOSTON_API" "$BUDGET_TASK_ID" running 10
sleep 3

budget_cost="$(state_field "$BOSTON_API" "$BUDGET_TASK_ID" accumulated_cost_usd)"
assert_float_gt "$budget_cost" 0.001 "budget-task accumulated cost"

http_code="$(
  curl -sS -o /tmp/magellan-budget-migrate.json -w '%{http_code}' -X POST \
    "${BOSTON_API}/tasks/${BUDGET_TASK_ID}/migrate/virginia"
)"
[[ "$http_code" == "409" ]] || {
  echo "Expected migration HTTP 409 after cost cap, got ${http_code}" >&2
  cat /tmp/magellan-budget-migrate.json >&2
  exit 1
}
python -m json.tool /tmp/magellan-budget-migrate.json
python - <<'PY'
import json
payload=json.load(open('/tmp/magellan-budget-migrate.json'))
detail=str(payload.get('detail','')).lower()
assert 'no feasible migration candidate' in detail, payload
print('cost cap pruned the migration candidate')
PY

curl -fsS -X POST "${BOSTON_API}/tasks/${BUDGET_TASK_ID}/stop" | python -m json.tool

echo "== Accounting ledger follows migration ownership Boston -> Virginia =="
curl -fsS -X POST "${BOSTON_API}/tasks/${MIGRATION_TASK_ID}/start" | python -m json.tool
wait_for_state "$BOSTON_API" "$MIGRATION_TASK_ID" running 10
sleep 3

boston_runtime="$(state_field "$BOSTON_API" "$MIGRATION_TASK_ID" accumulated_runtime_seconds)"
boston_cost="$(state_field "$BOSTON_API" "$MIGRATION_TASK_ID" accumulated_cost_usd)"
boston_carbon="$(state_field "$BOSTON_API" "$MIGRATION_TASK_ID" accumulated_carbon_grams)"
assert_float_gt "$boston_runtime" 0 "pre-migration runtime"
assert_float_gt "$boston_cost" 0 "pre-migration cost"
assert_float_gt "$boston_carbon" 0 "pre-migration carbon"

curl -fsS -X POST \
  "${BOSTON_API}/tasks/${MIGRATION_TASK_ID}/migrate/virginia" \
  | tee /tmp/magellan-accounting-migration.json \
  | python -m json.tool

python - <<'PY'
import json
payload=json.load(open('/tmp/magellan-accounting-migration.json'))
assert payload.get('migrated') is True, payload
assert payload['state']['owner_node_id'] == 'virginia', payload
print('migration completed')
PY

wait_for_state "$VIRGINIA_API" "$MIGRATION_TASK_ID" running 20

virginia_runtime="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_runtime_seconds)"
virginia_cost="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_cost_usd)"
virginia_carbon="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_carbon_grams)"
transfer_cost="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_transfer_cost_usd)"
transfer_carbon="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_transfer_carbon_grams)"
migration_seconds="$(state_field "$VIRGINIA_API" "$MIGRATION_TASK_ID" accumulated_migration_seconds)"

python - \
  "$virginia_runtime" "$boston_runtime" \
  "$virginia_cost" "$boston_cost" \
  "$virginia_carbon" "$boston_carbon" <<'PY'
import sys
vr,br,vc,bc,vcar,bcar=map(float,sys.argv[1:])
assert vr >= br, (vr,br)
assert vc >= bc, (vc,bc)
assert vcar >= bcar, (vcar,bcar)
print('owner handoff preserved cumulative runtime, cost, and carbon')
PY

assert_float_gt "$transfer_cost" 0 "transfer cost"
assert_float_gt "$transfer_carbon" 0 "transfer carbon"
assert_float_gt "$migration_seconds" 0 "migration downtime"

curl -fsS -X POST "${VIRGINIA_API}/tasks/${MIGRATION_TASK_ID}/stop" | python -m json.tool

trap - EXIT
cleanup

echo "ALL TWO-NODE PAUSE AND ACCOUNTING CHECKS PASSED"
