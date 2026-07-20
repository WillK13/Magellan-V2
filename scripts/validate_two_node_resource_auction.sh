#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
VIRGINIA_SSH="${VIRGINIA_SSH:-WILL@10.162.0.2}"
DEFINITION_FILE="${DEFINITION_FILE:-config/submissions/dev-counter-definition.json}"

json_field() {
  python -c 'import json,sys; value=json.load(sys.stdin); print(eval(sys.argv[1], {"value": value}))' "$1"
}

wait_bid() {
  local bid_id="$1" wanted="$2"
  for attempt in $(seq 1 30); do
    response="$(curl -fsS "$VIRGINIA_API/bids/$bid_id" 2>/dev/null || true)"
    if [[ -n "$response" ]]; then
      status="$(printf '%s' "$response" | json_field 'value["status"]')"
      echo "bid=$bid_id status=$status wanted=$wanted" >&2
      if [[ "$status" == "$wanted" ]]; then
        printf '%s' "$response"
        return 0
      fi
    fi
    sleep 1
  done
  echo "Timed out waiting for bid=$bid_id status=$wanted" >&2
  return 1
}

make_bid() {
  local output="$1" bid_id="$2" task_id="$3" score="$4" cpu="$5"
  python - "$bid_id" "$task_id" "$score" "$cpu" <<'PY' > "$output"
import json,sys
from datetime import datetime,timezone
bid_id,task_id,score,cpu=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
print(json.dumps({
  "bid_id": bid_id,
  "epoch_id": "resource-auction-validation",
  "task_id": task_id,
  "bidder_type": "task",
  "task_context": {
    "workload_type": "counter",
    "priority": 0,
    "deadline_at_utc": None,
    "estimated_remaining_seconds": 100,
    "checkpoint_bytes": 100,
    "static_data_bytes": 0,
    "accumulated_cost_usd": 0,
    "cost_cap_usd": None,
    "resource_request": {
      "cpu_cores": cpu,
      "memory_mb": 256,
      "gpu_count": 0,
      "accelerator_type": None
    },
    "fallback_action": "continue",
    "fallback_destination_node_id": None,
    "fallback_score": min(1.0, score + 0.2),
    "opportunity_loss": 0.2
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
    "score": score
  },
  "submitted_at_utc": datetime.now(timezone.utc).isoformat()
}))
PY
}

echo "== Peer and auction health =="
curl -fsS "$BOSTON_API/health" | python -m json.tool
curl -fsS "$VIRGINIA_API/health" | python -m json.tool
curl -fsS "$VIRGINIA_API/auction/status" \
  | tee /tmp/magellan-auction-status.json \
  | python -m json.tool
python - <<'PY'
import json
value=json.load(open('/tmp/magellan-auction-status.json'))
assert value['strategy']=='credit_fair', value
assert value['resource_capacity']['cpu_cores']==2, value
assert value['resource_capacity']['memory_mb']==4096, value
print('Virginia resource-aware credit auction is active')
PY

echo "== First window: base score wins and loser earns credit =="
A1="auction-a1-$RANDOM-$RANDOM"
B1="auction-b1-$RANDOM-$RANDOM"
make_bid /tmp/a1.json "$A1" task-score-winner 0.1 1
make_bid /tmp/b1.json "$B1" task-credit 0.8 1
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/a1.json "$VIRGINIA_API/bids" >/dev/null
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/b1.json "$VIRGINIA_API/bids" >/dev/null
wait_bid "$A1" accepted > /tmp/a1-result.json
wait_bid "$B1" rejected > /tmp/b1-result.json
python - <<'PY'
import json
winner=json.load(open('/tmp/a1-result.json'))
loser=json.load(open('/tmp/b1-result.json'))
assert winner['auction_strategy']=='credit_fair', winner
assert loser['auction_credit_before']==0, loser
assert loser['auction_credit_after']==1, loser
assert loser['resource_fit'] is True, loser
print('first loser earned one unit of destination credit')
PY
curl -fsS -X POST "$VIRGINIA_API/bids/$A1/cancel?reason=validation-release" >/dev/null

echo "== Second window: accumulated credit overrides lower score =="
A2="auction-a2-$RANDOM-$RANDOM"
B2="auction-b2-$RANDOM-$RANDOM"
make_bid /tmp/a2.json "$A2" task-score-winner 0.1 1
make_bid /tmp/b2.json "$B2" task-credit 0.8 1
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/a2.json "$VIRGINIA_API/bids" >/dev/null
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/b2.json "$VIRGINIA_API/bids" >/dev/null
wait_bid "$B2" accepted > /tmp/b2-result.json
wait_bid "$A2" rejected > /tmp/a2-result.json
python - <<'PY'
import json
credited=json.load(open('/tmp/b2-result.json'))
assert credited['auction_credit_before']==1, credited
assert credited['auction_credit_after']==0, credited
assert credited['candidate']['score']==0.8, credited
print('credit-fair policy selected the previously rejected task')
PY
curl -fsS -X POST "$VIRGINIA_API/bids/$B2/cancel?reason=validation-release" >/dev/null

echo "== Oversized resource request is infeasible and earns no credit =="
BIG="auction-big-$RANDOM-$RANDOM"
make_bid /tmp/big.json "$BIG" task-oversized 0.0 3
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/big.json "$VIRGINIA_API/bids" >/dev/null
wait_bid "$BIG" rejected > /tmp/big-result.json
python - <<'PY'
import json
value=json.load(open('/tmp/big-result.json'))
assert value['resource_fit'] is False, value
assert value['auction_credit_after']==0, value
assert 'CPU' in value['decision_reason'], value
print('oversized task was rejected without fairness credit')
PY

echo "== Dynamically submitted task produces a real resource-aware bid =="
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary "@$DEFINITION_FILE" "$BOSTON_API/task-definitions" >/tmp/resource-definition.json
python - <<'PY' > /tmp/resource-run.json
import json,uuid
print(json.dumps({
  "definition_id":"dynamic-counter",
  "revision":None,
  "initial_owner_node_id":"boston",
  "idempotency_key":"resource-auction-"+str(uuid.uuid4()),
  "auto_start":True,
  "labels":{"purpose":"resource-auction-validation"}
}))
PY
curl -fsS -X POST -H 'Content-Type: application/json' --data-binary @/tmp/resource-run.json "$BOSTON_API/task-runs" \
  | tee /tmp/resource-run-response.json \
  | python -m json.tool
RUN_ID="$(cat /tmp/resource-run-response.json | json_field 'value["run"]["run_id"]')"
sleep 3
curl -fsS -X POST "$BOSTON_API/tasks/$RUN_ID/migrate/virginia" \
  | tee /tmp/resource-migration.json \
  | python -m json.tool
python - <<'PY'
import json
value=json.load(open('/tmp/resource-migration.json'))
assert value['migrated'] is True, value
PY
sleep 2
curl -fsS "$VIRGINIA_API/bids" > /tmp/resource-bids.json
python - "$RUN_ID" <<'PY'
import json,sys
run_id=sys.argv[1]
bids=json.load(open('/tmp/resource-bids.json'))
matching=[b for b in bids if b['task_id']==run_id]
assert matching, bids
bid=matching[0]
assert bid['bidder_type']=='task', bid
assert bid['destination_node_id']=='virginia', bid
assert bid['status']=='consumed', bid
assert bid['resource_fit'] is True, bid
assert bid['task_context']['resource_request']['cpu_cores']==1, bid
assert bid['task_context']['fallback_score'] is not None, bid
assert bid['task_context']['opportunity_loss'] >= 0, bid
assert 'dominant_resource_share' in bid['auction_metrics'], bid
print('real dynamic task bid carried resources and fallback regret')
PY
curl -fsS -X POST "$VIRGINIA_API/tasks/$RUN_ID/stop" >/dev/null || true

echo "== Credits are durable on Virginia =="
ssh "$VIRGINIA_SSH" \
  'test -s ~/Magellan-V2/runtime-state-resource-auction/control/bids.json && echo VIRGINIA_AUCTION_STATE_OK'

echo "ALL TWO-NODE RESOURCE-AWARE AUCTION CHECKS PASSED"
