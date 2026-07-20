#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
COMMAND_DEFINITION="${COMMAND_DEFINITION:-config/submissions/dev-command-definition.json}"
DENDRO_DEFINITION="${DENDRO_DEFINITION:-config/submissions/dev-dendro-definition.json}"
STATE_ROOT_NAME="${STATE_ROOT_NAME:-runtime-state-generic-command-dendro}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

submit_run() {
  local definition_file="$1" definition_id="$2" purpose="$3" output="$4"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary "@$definition_file" \
    "$BOSTON_API/task-definitions" > /tmp/runtime-definition.json
  python -m json.tool /tmp/runtime-definition.json

  python - "$definition_id" "$purpose" <<'PY' > /tmp/runtime-run-request.json
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
PY

  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data-binary @/tmp/runtime-run-request.json \
    "$BOSTON_API/task-runs" > "$output"
  python -m json.tool "$output"
}

wait_bid_decision() {
  local bid_id="$1" output="$2"
  for attempt in $(seq 1 30); do
    response="$(curl -fsS "$VIRGINIA_API/bids/$bid_id" 2>/dev/null || true)"
    if [[ -n "$response" ]] && printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin)
raise SystemExit(0 if v["status"] != "pending" else 1)
' 2>/dev/null; then
      printf '%s' "$response" > "$output"
      return 0
    fi
    echo "waiting for bid decision bid=$bid_id attempt=$attempt" >&2
    sleep 1
  done
  return 1
}

wait_task_telemetry() {
  local api="$1" task_id="$2" minimum_processes="$3" output="$4"
  for attempt in $(seq 1 60); do
    response="$(curl -fsS "$api/telemetry/tasks/$task_id" 2>/dev/null || true)"
    if [[ -n "$response" ]] && printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin); minimum=int(sys.argv[1])
ok=(v["freshness"]=="fresh" and v["sample_count"]>=1 and
    v["process_count"]>=minimum and (v["checkpoint_bytes"] or 0)>0)
raise SystemExit(0 if ok else 1)
' "$minimum_processes" 2>/dev/null; then
      printf '%s' "$response" > "$output"
      return 0
    fi
    echo "waiting for task telemetry task=$task_id attempt=$attempt" >&2
    sleep 1
  done
  return 1
}

wait_progress_greater() {
  local api="$1" task_id="$2" minimum="$3" output="$4"
  for attempt in $(seq 1 60); do
    response="$(curl -fsS "$api/task-runs/$task_id" 2>/dev/null || true)"
    if [[ -n "$response" ]] && printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin); minimum=float(sys.argv[1])
progress=v["state"].get("progress_completed_units")
ok=(progress is not None and progress > minimum)
raise SystemExit(0 if ok else 1)
' "$minimum" 2>/dev/null; then
      printf '%s' "$response" > "$output"
      return 0
    fi
    echo "waiting for resumed progress task=$task_id attempt=$attempt" >&2
    sleep 1
  done
  return 1
}

echo "== Runtime capability discovery =="
curl -fsS "$BOSTON_API/capabilities" \
  | tee /tmp/runtime-boston-capabilities.json \
  | python -m json.tool
curl -fsS "$VIRGINIA_API/capabilities" \
  | tee /tmp/runtime-virginia-capabilities.json \
  | python -m json.tool
python - <<'PY'
import json
for path in (
    "/tmp/runtime-boston-capabilities.json",
    "/tmp/runtime-virginia-capabilities.json",
):
    value=json.load(open(path))
    assert value["ready"] is True, value
    assert value["configured"]["architecture"] == "x86_64", value
    assert value["observed"]["architecture"] == "x86_64", value
    assert "command" in value["runtime_adapters"], value
    assert "dendro" in value["runtime_adapters"], value
    assert "python3" in value["observed"]["commands"], value
print("runtime capability discovery passed")
PY

echo "== Destination rejects an incompatible task bid =="
python - <<'PY' > /tmp/runtime-incompatible-bid.json
import json
import uuid
from datetime import datetime, timezone
bid_id="incompatible-"+str(uuid.uuid4())
print(json.dumps({
    "bid_id": bid_id,
    "epoch_id": "compatibility-validation",
    "task_id": "arm-only-dendro-task",
    "bidder_type": "task",
    "task_context": {
        "workload_type": "dendro-gr",
        "resource_request": {
            "cpu_cores": 1,
            "memory_mb": 256,
            "gpu_count": 0,
            "accelerator_type": None,
        },
        "compatibility": {
            "architectures": ["aarch64"],
            "operating_systems": ["linux"],
            "required_commands": ["python3"],
            "required_features": ["dendro-adapter"],
        },
    },
    "source_node_id": "boston",
    "destination_node_id": "virginia",
    "candidate": {
        "action": "migrate",
        "source_node_id": "boston",
        "destination_node_id": "virginia",
        "time_seconds": 1,
        "carbon_grams": 1,
        "cost_usd": 0.01,
        "details": {},
        "normalized_time": 0,
        "normalized_carbon": 0,
        "normalized_cost": 0,
        "score": 0.1,
    },
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
}))
PY
curl -fsS -X POST -H 'Content-Type: application/json' \
  --data-binary @/tmp/runtime-incompatible-bid.json \
  "$VIRGINIA_API/bids" > /tmp/runtime-incompatible-submit.json
BID_ID="$(cat /tmp/runtime-incompatible-bid.json | json_field 'value["bid_id"]')"
wait_bid_decision "$BID_ID" /tmp/runtime-incompatible-result.json
python -m json.tool /tmp/runtime-incompatible-result.json
python - <<'PY'
import json
v=json.load(open('/tmp/runtime-incompatible-result.json'))
assert v['status']=='rejected', v
assert v['compatibility_fit'] is False, v
assert any('architecture' in item for item in v['compatibility_reasons']), v
assert v['auction_credit_after']==0, v
print('hard compatibility rejection passed')
PY

echo "== Generic command adapter =="
submit_run \
  "$COMMAND_DEFINITION" \
  generic-command-counter \
  generic-command-validation \
  /tmp/runtime-command-run.json
COMMAND_RUN="$(cat /tmp/runtime-command-run.json | json_field 'value["run"]["run_id"]')"
python - <<'PY'
import json
v=json.load(open('/tmp/runtime-command-run.json'))
s=v['state']
assert s['status']=='running', v
assert s['runtime_adapter']=='command', v
assert s['process_group_id']==s['pid'], v
assert s['launch_command'][0]=='python3', v
print('generic command launch metadata passed')
PY
curl -fsS -X POST "$BOSTON_API/tasks/$COMMAND_RUN/stop" | python -m json.tool

echo "== Dendro-compatible MPI process tree and checkpoint =="
submit_run \
  "$DENDRO_DEFINITION" \
  dendro-bssn-validation \
  dendro-runtime-validation \
  /tmp/runtime-dendro-run.json
DENDRO_RUN="$(cat /tmp/runtime-dendro-run.json | json_field 'value["run"]["run_id"]')"
SOURCE_PID="$(cat /tmp/runtime-dendro-run.json | json_field 'value["state"]["pid"]')"
echo "dendro_run=$DENDRO_RUN source_process_group=$SOURCE_PID"
wait_task_telemetry "$BOSTON_API" "$DENDRO_RUN" 3 /tmp/runtime-dendro-source-telemetry.json
python -m json.tool /tmp/runtime-dendro-source-telemetry.json
curl -fsS "$BOSTON_API/task-runs/$DENDRO_RUN" > /tmp/runtime-dendro-source-state.json
SOURCE_STEP="$(cat /tmp/runtime-dendro-source-state.json | json_field 'value["state"]["progress_completed_units"]')"
python - "$DENDRO_RUN" "$STATE_ROOT_NAME" <<'PY'
import json
import pathlib
import sys
run_id=sys.argv[1]
root=pathlib.Path(sys.argv[2])
manifest=root/'tasks'/run_id/'checkpoint'/'manifest.json'
assert manifest.is_file(), manifest
value=json.loads(manifest.read_text())
assert value['world_size']==2, value
assert len(value['files'])==3, value
for item in value['files']:
    path=manifest.parent/item['path']
    assert path.is_file(), path
    assert path.stat().st_size==item['size_bytes'], (path,item)
print('manifest-based Dendro checkpoint passed')
PY

echo "== Migrate Dendro Boston -> Virginia =="
curl -fsS -X POST "$BOSTON_API/tasks/$DENDRO_RUN/migrate/virginia" \
  | tee /tmp/runtime-dendro-migration.json \
  | python -m json.tool
python - <<'PY'
import json
v=json.load(open('/tmp/runtime-dendro-migration.json'))
assert v['migrated'] is True, v
print('Dendro migration completed')
PY
sleep 1
if pgrep -g "$SOURCE_PID" >/dev/null 2>&1; then
  echo "Boston Dendro process group still exists: $SOURCE_PID" >&2
  pgrep -a -g "$SOURCE_PID" >&2 || true
  exit 1
fi
echo "all Boston launcher/rank processes stopped"

wait_task_telemetry "$VIRGINIA_API" "$DENDRO_RUN" 3 /tmp/runtime-dendro-destination-telemetry.json
python -m json.tool /tmp/runtime-dendro-destination-telemetry.json
wait_progress_greater "$VIRGINIA_API" "$DENDRO_RUN" "$SOURCE_STEP" /tmp/runtime-dendro-destination-state.json
python -m json.tool /tmp/runtime-dendro-destination-state.json
python - <<'PY'
import json
source=json.load(open('/tmp/runtime-dendro-source-state.json'))['state']
destination=json.load(open('/tmp/runtime-dendro-destination-state.json'))['state']
assert destination['owner_node_id']=='virginia', destination
assert destination['generation']==source['generation']+1, (source,destination)
assert destination['runtime_adapter']=='dendro', destination
assert destination['resumed_from_checkpoint'] is True, destination
assert '--resume' in destination['launch_command'], destination
assert destination['progress_completed_units']>source['progress_completed_units'], (source,destination)
print('Dendro resumed from transferred checkpoint and advanced progress')
PY

ssh "$VIRGINIA_SSH" "
cd ~/Magellan-V2
.venv/bin/python - <<'PY'
import json
from pathlib import Path
run_id='$DENDRO_RUN'
root=Path('$STATE_ROOT_NAME')
state=json.loads((root/'tasks'/run_id/'state.json').read_text())
manifest=root/'tasks'/run_id/'checkpoint'/'manifest.json'
assert state['runtime_adapter']=='dendro', state
assert state['resumed_from_checkpoint'] is True, state
assert manifest.is_file(), manifest
print('VIRGINIA_DURABLE_DENDRO_STATE_OK')
PY
"

curl -fsS -X POST "$VIRGINIA_API/tasks/$DENDRO_RUN/stop" | python -m json.tool

echo "ALL TWO-NODE GENERIC COMMAND AND DENDRO CHECKS PASSED"
