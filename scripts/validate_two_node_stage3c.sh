#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
ROOT="${MAGELLAN_STATE_ROOT:-$HOME/Magellan-V2/runtime-state-gcp-measurement}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

wait_bid_terminal() {
  local bid_id="$1"
  for _ in $(seq 1 45); do
    response="$(curl -fsS "$VIRGINIA_API/bids/$bid_id" 2>/dev/null || true)"
    if [[ -n "$response" ]]; then
      status="$(printf '%s' "$response" | json_field 'value["status"]')"
      if [[ "$status" != "pending" ]]; then
        printf '%s' "$response"
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for bid $bid_id" >&2
  return 1
}

make_bid() {
  local output="$1" bid_id="$2"
  python - "$bid_id" <<'PY' > "$output"
import json,sys
from datetime import datetime,timezone
bid_id=sys.argv[1]
print(json.dumps({
  "bid_id": bid_id,
  "epoch_id": "stage3c-resource-validation",
  "task_id": "stage3c-task-" + bid_id,
  "bidder_type": "task",
  "task_context": {
    "workload_type": "benchmark-matmul",
    "estimated_remaining_seconds": 600,
    "checkpoint_bytes": 4096,
    "static_data_bytes": 0,
    "resource_request": {
      "cpu_cores": 1.0,
      "memory_mb": 256,
      "gpu_count": 0,
      "accelerator_type": None
    }
  },
  "source_node_id": "boston",
  "destination_node_id": "virginia",
  "candidate": {
    "action": "migrate",
    "source_node_id": "boston",
    "destination_node_id": "virginia",
    "time_seconds": 1,
    "carbon_grams": 1,
    "cost_usd": 0,
    "details": {},
    "normalized_time": 0,
    "normalized_carbon": 0,
    "normalized_cost": 0,
    "score": 0.1
  },
  "submitted_at_utc": datetime.now(timezone.utc).isoformat()
}))
PY
}

echo "== Stage 3C resource-only destination admission =="
curl -fsS "$VIRGINIA_API/auction/status" \
  | tee /tmp/stage3c-auction-before.json \
  | python -m json.tool
python - <<'PY'
import json
value=json.load(open('/tmp/stage3c-auction-before.json'))
assert value['task_slot_capacity'] is None, value
assert value['resource_capacity']['cpu_cores']==2, value
assert value['resource_capacity']['memory_mb']==16384, value
assert value['reserved_cpu_cores']==0, (
    'Virginia must start validation with no reserved/running CPU', value
)
assert value['available_cpu_cores']==2, value
print('RESOURCE_ONLY_GCP_CONFIG_PASS')
PY

BIDS=()
for suffix in a b c; do
  bid_id="stage3c-${suffix}-$RANDOM-$RANDOM"
  BIDS+=("$bid_id")
  make_bid "/tmp/${bid_id}.json" "$bid_id"
  curl -fsS -X POST \
    -H 'Content-Type: application/json' \
    --data-binary "@/tmp/${bid_id}.json" \
    "$VIRGINIA_API/bids" >/dev/null
done

for index in 0 1 2; do
  wait_bid_terminal "${BIDS[$index]}" > "/tmp/stage3c-bid-${index}.json"
done

python - <<'PY'
import json
records=[json.load(open(f'/tmp/stage3c-bid-{i}.json')) for i in range(3)]
accepted=[r for r in records if r['status']=='accepted']
rejected=[r for r in records if r['status']=='rejected']
assert len(accepted)==2, records
assert len(rejected)==1, records
assert rejected[0]['resource_fit'] is False, rejected[0]
assert 'CPU' in (rejected[0]['decision_reason'] or ''), rejected[0]
print('RESOURCE_VECTOR_CONTENTION_PASS')
PY

curl -fsS "$VIRGINIA_API/auction/status" \
  | tee /tmp/stage3c-auction-full.json \
  | python -m json.tool
python - <<'PY'
import json
value=json.load(open('/tmp/stage3c-auction-full.json'))
assert value['available_cpu_cores']==0, value
assert value['reserved_cpu_cores']==2, value
assert abs(value['resource_busy_fraction']-1.0)<1e-9, value
print('RESOURCE_BUSY_FRACTION_PASS')
PY

for index in 0 1 2; do
  status="$(cat "/tmp/stage3c-bid-${index}.json" | json_field 'value["status"]')"
  if [[ "$status" == "accepted" ]]; then
    curl -fsS -X POST \
      "$VIRGINIA_API/bids/${BIDS[$index]}/cancel?reason=stage3c-validation-release" \
      >/dev/null
  fi
done

sleep 1
curl -fsS "$VIRGINIA_API/auction/status" > /tmp/stage3c-auction-released.json
python - <<'PY'
import json
value=json.load(open('/tmp/stage3c-auction-released.json'))
assert value['available_cpu_cores']==2, value
assert value['reserved_cpu_cores']==0, value
assert abs(value['resource_busy_fraction'])<1e-9, value
print('RESOURCE_RELEASE_PASS')
PY

echo "== Seeded workload population + real checkpointable benchmark =="
POP_ID="stage3c-gcp-$RANDOM-$RANDOM"
POP_DIR="/tmp/$POP_ID"
python scripts/populate_workload.py \
  --cluster config/cluster.gcp.json \
  --count 1 \
  --seed 314159 \
  --mix matmul=1 \
  --population-id "$POP_ID" \
  --output "$POP_DIR" \
  --initial-nodes boston \
  --benchmark-iterations 10000 \
  --submit \
  --start

RUN_ID="$(python - "$POP_DIR/submitted.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
assert len(value)==1, value
print(value[0]['run_id'])
PY
)"
echo "RUN_ID=$RUN_ID"
TASKDIR="$ROOT/tasks/$RUN_ID"

for attempt in $(seq 1 30); do
  if [[ -s "$TASKDIR/checkpoint/benchmark.json" && -s "$TASKDIR/runtime/progress.json" ]]; then
    break
  fi
  sleep 1
done

test -s "$TASKDIR/checkpoint/benchmark.json"
test -s "$TASKDIR/runtime/progress.json"
python - "$TASKDIR" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
checkpoint=json.loads((root/'checkpoint/benchmark.json').read_text())
progress=json.loads((root/'runtime/progress.json').read_text())
assert checkpoint['benchmark']=='matmul', checkpoint
assert checkpoint['completed_iterations']>0, checkpoint
assert progress['completed_units']>0, progress
assert progress['total_units']==10000, progress
print('REAL_CHECKPOINTABLE_BENCHMARK_PASS', checkpoint['completed_iterations'])
PY

curl -fsS "$BOSTON_API/task-runs/$RUN_ID" | python -m json.tool
curl -fsS -X POST "$BOSTON_API/task-runs/$RUN_ID/stop" >/dev/null || true

echo "ALL TWO-NODE STAGE 3C CHECKS PASSED"
