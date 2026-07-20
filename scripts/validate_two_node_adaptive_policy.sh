#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
PRESSURE_DEFINITION="${PRESSURE_DEFINITION:-config/submissions/dev-adaptive-pressure-definition.json}"
MIGRATE_DEFINITION="${MIGRATE_DEFINITION:-config/submissions/dev-adaptive-migrate-definition.json}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

submit_run() {
  local definition_file="$1" definition_id="$2" purpose="$3" output="$4"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "@$definition_file" \
    "$BOSTON_API/task-definitions" > /tmp/adaptive-definition.json
  python -m json.tool /tmp/adaptive-definition.json

  python - "$definition_id" "$purpose" <<'PYCODE' > /tmp/adaptive-run-request.json
import json
import sys
import uuid

print(json.dumps({
    "definition_id": sys.argv[1],
    "revision": None,
    "initial_owner_node_id": "boston",
    "idempotency_key": sys.argv[2] + "-" + str(uuid.uuid4()),
    "auto_start": True,
    "labels": {"purpose": sys.argv[2]},
}))
PYCODE

  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary @/tmp/adaptive-run-request.json \
    "$BOSTON_API/task-runs" > "$output"
  python -m json.tool "$output"
}

wait_policy_state() {
  local api="$1" task_id="$2" minimum_decisions="$3" output="$4"
  for attempt in $(seq 1 75); do
    response="$(curl -fsS "$api/policy/tasks/$task_id" 2>/dev/null || true)"
    if [[ -n "$response" ]] && printf '%s' "$response" | python -c '
import json
import sys

value = json.load(sys.stdin)
minimum = int(sys.argv[1])
ok = value["decision_count"] >= minimum and value["last_decision"] is not None
raise SystemExit(0 if ok else 1)
' "$minimum_decisions" 2>/dev/null; then
      printf '%s' "$response" > "$output"
      return 0
    fi
    echo "waiting for adaptive policy task=$task_id attempt=$attempt" >&2
    sleep 1
  done
  echo "Timed out waiting for adaptive policy: $task_id" >&2
  return 1
}

wait_owner() {
  local api="$1" task_id="$2" owner="$3"
  for attempt in $(seq 1 60); do
    actual="$(curl -fsS "$api/tasks" | python -c '
import json
import sys

value = json.load(sys.stdin)
for item in value.get("tasks", []):
    if item["state"]["task_id"] == sys.argv[1]:
        print(item["state"]["owner_node_id"])
        break
' "$task_id" 2>/dev/null || true)"
    echo "task=$task_id owner=$actual wanted=$owner" >&2
    [[ "$actual" == "$owner" ]] && return 0
    sleep 1
  done
  return 1
}

echo "== Health and adaptive policy configuration =="
curl -fsS "$BOSTON_API/health" | tee /tmp/adaptive-boston-health.json | python -m json.tool
curl -fsS "$VIRGINIA_API/health" | tee /tmp/adaptive-virginia-health.json | python -m json.tool
curl -fsS "$BOSTON_API/policy" | tee /tmp/adaptive-policy-summary.json | python -m json.tool
python - <<'PYCODE'
import json

for path in [
    "/tmp/adaptive-boston-health.json",
    "/tmp/adaptive-virginia-health.json",
]:
    value = json.load(open(path))
    assert value["status"] == "ok", value
    assert value["adaptive_policy_enabled"] is True, value
summary = json.load(open("/tmp/adaptive-policy-summary.json"))
assert summary["adaptive"]["multiplier_bound_fraction"] == 0.25, summary
assert summary["baseline_weights"] == {
    "time": 0.25,
    "carbon": 0.5,
    "cost": 0.25,
}, summary
print("adaptive policy configuration passed")
PYCODE

echo "== Offline baseline calibration utility =="
python -m magellan.policy.calibration \
  --input config/policy-calibration.example.json \
  --cost-cap-usd 10 \
  --deadline-seconds 100000 \
  --output /tmp/adaptive-calibration-result.json
python -m json.tool /tmp/adaptive-calibration-result.json
python - <<'PYCODE'
import json

value = json.load(open("/tmp/adaptive-calibration-result.json"))
assert value["feasible_candidate_count"] >= 1, value
weights = value["selected"]["candidate"]["weights"]
assert abs(sum(weights.values()) - 1) < 1e-9, value
print("offline baseline calibration passed")
PYCODE

echo "== Budget and deadline pressure adapt effective weights =="
submit_run \
  "$PRESSURE_DEFINITION" \
  adaptive-pressure-counter \
  adaptive-pressure \
  /tmp/adaptive-pressure-run.json
PRESSURE_RUN="$(cat /tmp/adaptive-pressure-run.json | json_field 'value["run"]["run_id"]')"
wait_policy_state \
  "$BOSTON_API" \
  "$PRESSURE_RUN" \
  1 \
  /tmp/adaptive-pressure-policy.json
python -m json.tool /tmp/adaptive-pressure-policy.json
python - <<'PYCODE'
import json

value = json.load(open("/tmp/adaptive-pressure-policy.json"))
baseline = value["baseline_weights"]
effective = value["effective_weights"]
mult = value["multipliers"]
record = value["last_decision"]
assert abs(sum(effective.values()) - 1) < 1e-9, value
assert mult["time"] == 1.25, value
assert mult["cost"] > 1.0, value
assert all(0.75 <= item <= 1.25 for item in mult.values()), value
assert effective["time"] > baseline["time"], value
assert effective["cost"] > baseline["cost"], value
assert record["signals"]["deadline_at_risk"] is True, value
assert record["hard_constraints"]["cost_cap_enabled"] is True, value
assert record["hard_constraints"]["cost_cap_pruned_migrations"] >= 1, value
bounds = record["normalization_bounds"]
assert bounds["time_max"] >= bounds["time_min"], value
print("bounded budget/deadline adaptation and hard constraints passed")
PYCODE
curl -fsS -X POST "$BOSTON_API/tasks/$PRESSURE_RUN/stop" | python -m json.tool

echo "== Adaptive state follows migration Boston -> Virginia =="
submit_run \
  "$MIGRATE_DEFINITION" \
  adaptive-migrate-counter \
  adaptive-migrate \
  /tmp/adaptive-migrate-run.json
MIGRATE_RUN="$(cat /tmp/adaptive-migrate-run.json | json_field 'value["run"]["run_id"]')"
wait_policy_state \
  "$BOSTON_API" \
  "$MIGRATE_RUN" \
  1 \
  /tmp/adaptive-source-policy.json
SOURCE_COUNT="$(cat /tmp/adaptive-source-policy.json | json_field 'value["decision_count"]')"
python -m json.tool /tmp/adaptive-source-policy.json
curl -fsS -X POST "$BOSTON_API/tasks/$MIGRATE_RUN/migrate/virginia" \
  | tee /tmp/adaptive-migration.json \
  | python -m json.tool
python - <<'PYCODE'
import json

value = json.load(open("/tmp/adaptive-migration.json"))
assert value["migrated"] is True, value
print("migration completed")
PYCODE
wait_owner "$BOSTON_API" "$MIGRATE_RUN" virginia
wait_policy_state \
  "$VIRGINIA_API" \
  "$MIGRATE_RUN" \
  "$SOURCE_COUNT" \
  /tmp/adaptive-destination-policy.json
python -m json.tool /tmp/adaptive-destination-policy.json
python - <<'PYCODE'
import json

source = json.load(open("/tmp/adaptive-source-policy.json"))
destination = json.load(open("/tmp/adaptive-destination-policy.json"))
assert destination["decision_count"] >= source["decision_count"], (
    source,
    destination,
)
assert destination["baseline_weights"] == source["baseline_weights"], (
    source,
    destination,
)
assert destination["normalization"]["time"]["samples"], destination
print("adaptive decision count and rolling normalization followed ownership")
PYCODE

curl -fsS "$VIRGINIA_API/ownership/snapshot" > /tmp/adaptive-ownership.json
python - "$MIGRATE_RUN" <<'PYCODE'
import json
import sys

snapshot = json.load(open("/tmp/adaptive-ownership.json"))
match = next(
    item for item in snapshot["updates"] if item["task_id"] == sys.argv[1]
)
assert match["owner_node_id"] == "virginia", match
assert match["adaptive_policy"] is not None, match
assert match["adaptive_policy"]["decision_count"] >= 1, match
print("ownership anti-entropy includes adaptive state")
PYCODE

curl -fsS -X POST "$VIRGINIA_API/tasks/$MIGRATE_RUN/stop" | python -m json.tool

echo "== Durable policy files =="
test -s runtime-state-adaptive-policy/control/adaptive-policy.json
ssh "$VIRGINIA_SSH" \
  'test -s ~/Magellan-V2/runtime-state-adaptive-policy/control/adaptive-policy.json && echo VIRGINIA_ADAPTIVE_POLICY_STATE_OK'

python -m json.tool runtime-state-adaptive-policy/control/adaptive-policy.json \
  | sed -n '1,120p'

echo "ALL TWO-NODE ADAPTIVE POLICY CHECKS PASSED"
