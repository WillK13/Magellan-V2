#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
DEFINITION_FILE="${DEFINITION_FILE:-config/submissions/dev-telemetry-counter-definition.json}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

wait_task_telemetry() {
  local api="$1" task_id="$2" output="$3"
  for attempt in $(seq 1 30); do
    response="$(curl -fsS "$api/telemetry/tasks/$task_id" 2>/dev/null || true)"
    if [[ -n "$response" ]]; then
      if printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin)
ok=(v["freshness"]=="fresh" and v["sample_count"]>=2 and
    v["cpu_utilization_percent"] is not None and
    (v["memory_rss_mb"] or 0)>0 and (v["checkpoint_bytes"] or 0)>0 and
    v["effective_power_kw"]>0)
raise SystemExit(0 if ok else 1)
' 2>/dev/null; then
        printf '%s' "$response" > "$output"
        return 0
      fi
    fi
    echo "waiting for fresh task telemetry task=$task_id attempt=$attempt" >&2
    sleep 1
  done
  echo "Timed out waiting for task telemetry: $task_id" >&2
  return 1
}

wait_edge_latency() {
  local output="$1"
  for attempt in $(seq 1 30); do
    response="$(curl -fsS "$BOSTON_API/telemetry/edges/virginia" 2>/dev/null || true)"
    if [[ -n "$response" ]] && printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin)
raise SystemExit(0 if v["latency_freshness"]=="fresh" and v["latency_source"]=="measured_http_rtt" else 1)
' 2>/dev/null; then
      printf '%s' "$response" > "$output"
      return 0
    fi
    echo "waiting for Boston->Virginia latency telemetry attempt=$attempt" >&2
    sleep 1
  done
  return 1
}

wait_transfer_calibration() {
  for attempt in $(seq 1 30); do
    curl -fsS "$BOSTON_API/telemetry/edges/virginia" > /tmp/telemetry-edge-after.json
    curl -fsS "$BOSTON_API/telemetry/calibration" > /tmp/telemetry-calibration.json
    if python - <<'PY' 2>/dev/null
import json
edge=json.load(open('/tmp/telemetry-edge-after.json'))
cal=json.load(open('/tmp/telemetry-calibration.json'))
matching=[x for x in cal if x['source_node_id']=='boston' and x['destination_node_id']=='virginia']
ok=(edge['bandwidth_freshness']=='fresh' and
    edge['bandwidth_source']=='measured_transfer_ema' and
    edge['effective_bandwidth_mbps']>0 and matching and
    matching[0]['freshness']=='fresh' and matching[0]['sample_count']>=1 and
    matching[0]['checkpoint_seconds_ema'] is not None and
    matching[0]['restore_seconds_ema'] is not None and
    matching[0]['total_downtime_seconds_ema'] is not None)
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    echo "waiting for transfer bandwidth and migration calibration attempt=$attempt" >&2
    sleep 1
  done
  return 1
}

wait_task_status() {
  local api="$1" task_id="$2" wanted="$3"
  for attempt in $(seq 1 90); do
    response="$(curl -fsS "$api/tasks" 2>/dev/null || true)"
    status="$(printf '%s' "$response" | python -c '
import json,sys
v=json.load(sys.stdin); tid=sys.argv[1]
for item in v.get("tasks",[]):
    if item["state"]["task_id"]==tid:
        print(item["state"]["status"]); break
' "$task_id" 2>/dev/null || true)"
    echo "task=$task_id status=$status wanted=$wanted" >&2
    [[ "$status" == "$wanted" ]] && return 0
    sleep 1
  done
  return 1
}

echo "== Peer health and telemetry service =="
curl -fsS "$BOSTON_API/health" | tee /tmp/telemetry-boston-health.json | python -m json.tool
curl -fsS "$VIRGINIA_API/health" | tee /tmp/telemetry-virginia-health.json | python -m json.tool
python - <<'PY'
import json
for path in ['/tmp/telemetry-boston-health.json','/tmp/telemetry-virginia-health.json']:
    value=json.load(open(path))
    assert value['status']=='ok', value
    assert 'telemetry_state_file' in value, value
print('telemetry background services are active')
PY

wait_edge_latency /tmp/telemetry-edge-before.json
python -m json.tool /tmp/telemetry-edge-before.json
python - <<'PY'
import json
v=json.load(open('/tmp/telemetry-edge-before.json'))
assert v['latency_source']=='measured_http_rtt', v
assert v['latency_freshness']=='fresh', v
assert v['effective_latency_ms']>=0, v
print('live peer latency measurement is fresh')
PY

echo "== Submit and start a runtime task with no static task files =="
curl -fsS -X POST -H 'Content-Type: application/json' \
  --data-binary "@$DEFINITION_FILE" \
  "$BOSTON_API/task-definitions" \
  | tee /tmp/telemetry-definition.json \
  | python -m json.tool
python - <<'PY' > /tmp/telemetry-run.json
import json,uuid
print(json.dumps({
  'definition_id':'telemetry-counter',
  'revision':None,
  'initial_owner_node_id':'boston',
  'idempotency_key':'telemetry-validation-'+str(uuid.uuid4()),
  'auto_start':True,
  'labels':{'purpose':'telemetry-live-models-validation'}
}))
PY
curl -fsS -X POST -H 'Content-Type: application/json' \
  --data-binary @/tmp/telemetry-run.json \
  "$BOSTON_API/task-runs" \
  | tee /tmp/telemetry-run-response.json \
  | python -m json.tool
RUN_ID="$(cat /tmp/telemetry-run-response.json | json_field 'value["run"]["run_id"]')"
echo "run_id=$RUN_ID"

wait_task_telemetry "$BOSTON_API" "$RUN_ID" /tmp/telemetry-task-boston.json
python -m json.tool /tmp/telemetry-task-boston.json
python - <<'PY'
import json
v=json.load(open('/tmp/telemetry-task-boston.json'))
assert v['node_id']=='boston', v
assert v['process_count']>=1, v
assert v['memory_rss_mb']>0, v
assert v['checkpoint_bytes']>0, v
assert v['power_source'] in {'procfs_cpu_utilization_model','rapl_cpu_share'}, v
assert v['power_confidence']>=0.75, v
print('live task CPU, RSS, checkpoint, and power telemetry passed')
PY

echo "== Migrate and learn real transfer/restore behavior =="
curl -fsS -X POST "$BOSTON_API/tasks/$RUN_ID/migrate/virginia" \
  | tee /tmp/telemetry-migration.json \
  | python -m json.tool
python - <<'PY'
import json
v=json.load(open('/tmp/telemetry-migration.json'))
assert v['migrated'] is True, v
print('real migration completed')
PY

wait_transfer_calibration
python -m json.tool /tmp/telemetry-edge-after.json
python -m json.tool /tmp/telemetry-calibration.json
python - <<'PY'
import json
edge=json.load(open('/tmp/telemetry-edge-after.json'))
cal=json.load(open('/tmp/telemetry-calibration.json'))
record=next(x for x in cal if x['source_node_id']=='boston' and x['destination_node_id']=='virginia')
assert edge['bandwidth_source']=='measured_transfer_ema', edge
assert record['freshness']=='fresh', record
assert record['checkpoint_seconds_ema']>=0, record
assert record['transfer_seconds_ema']>0, record
assert record['restore_seconds_ema']>=0, record
assert record['total_downtime_seconds_ema']>0, record
print('measured bandwidth and migration phase calibration passed')
PY

wait_task_telemetry "$VIRGINIA_API" "$RUN_ID" /tmp/telemetry-task-virginia.json
python -m json.tool /tmp/telemetry-task-virginia.json
python - <<'PY'
import json
v=json.load(open('/tmp/telemetry-task-virginia.json'))
assert v['node_id']=='virginia', v
assert v['freshness']=='fresh', v
assert v['memory_rss_mb']>0, v
print('destination task telemetry started after migration')
PY

wait_task_status "$VIRGINIA_API" "$RUN_ID" completed

echo "== Durable telemetry state exists on both nodes =="
test -s runtime-state-telemetry/control/telemetry.json
ssh "$VIRGINIA_SSH" \
  'test -s ~/Magellan-V2/runtime-state-telemetry/control/telemetry.json && echo VIRGINIA_TELEMETRY_STATE_OK'

python -m json.tool runtime-state-telemetry/control/telemetry.json > /tmp/boston-telemetry-pretty.json
head -80 /tmp/boston-telemetry-pretty.json

echo "ALL TWO-NODE TELEMETRY LIVE-MODEL CHECKS PASSED"
