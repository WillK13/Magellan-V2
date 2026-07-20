#!/usr/bin/env bash
set -euo pipefail

BOSTON_API="${BOSTON_API:-http://127.0.0.1:8040}"
VIRGINIA_API="${VIRGINIA_API:-http://10.162.0.2:8040}"
STATE_ROOT_NAME="${STATE_ROOT_NAME:-runtime-state-v1-parity-closeout}"

echo "== Closeout health and policy surface =="
for API in "$BOSTON_API" "$VIRGINIA_API"; do
  curl -fsS "$API/health" | python -c '
import json,sys
value=json.load(sys.stdin)
assert value["status"] == "ok", value
assert value["carbon_forecast_enabled"] is True, value
assert value["carbon_forecast_provider"] == "linear_trend", value
assert len(value["pause_candidate_idle_seconds"]) >= 3, value
print(json.dumps({
    "node_id": value["node_id"],
    "forecast": value["carbon_forecast_provider"],
    "pause_candidates": value["pause_candidate_idle_seconds"],
}, indent=2))
'
done

echo "== History-only carbon forecast API =="
for NODE in boston virginia; do
  API="$BOSTON_API"
  [[ "$NODE" == "virginia" ]] && API="$VIRGINIA_API"
  curl -fsS "$API/carbon/forecast/$NODE?horizon_seconds=3600" \
    | tee "/tmp/${NODE}-forecast.json" \
    | python -m json.tool
  python - "$NODE" <<'PY'
import json,sys
value=json.load(open(f"/tmp/{sys.argv[1]}-forecast.json"))
assert value["average_g_per_kwh"] >= 0, value
assert value["source"] in {
    "linear_trend", "persistence", "configured_fallback"
}, value
assert 0 <= value["confidence"] <= 1, value
assert value["history_points"] >= 0, value
print(f"{sys.argv[1]} forecast passed")
PY
done

echo "== Pause search and migration-carbon unit contracts =="
python -m pytest -q \
  tests/test_pause_candidate_search.py \
  tests/test_migration_carbon_split.py \
  tests/test_carbon_forecast.py

echo "== Complete calibration grid and policy writer =="
python -m magellan.policy.calibration \
  --step-size 0.5 \
  --generate-grid-output /tmp/magellan-weight-grid.json
python - <<'PY'
import json
value=json.load(open('/tmp/magellan-weight-grid.json'))
assert value['step_size'] == 0.5, value
assert len(value['weights']) == 6, value
assert all(abs(sum(item.values()) - 1) < 1e-12 for item in value['weights'])
print('simplex grid generation passed')
PY
python -m magellan.policy.calibration \
  --input config/policy-calibration.example.json \
  --cost-cap-usd 10 \
  --deadline-seconds 100000 \
  --output /tmp/magellan-calibration-result.json \
  --policy-template config/policy.dev.json \
  --policy-output /tmp/magellan-calibrated-policy.json
python - <<'PY'
import json
result=json.load(open('/tmp/magellan-calibration-result.json'))
policy=json.load(open('/tmp/magellan-calibrated-policy.json'))
selected=result['selected']['candidate']['weights']
assert policy['weights'] == selected, (policy['weights'], selected)
assert abs(sum(policy['weights'].values()) - 1) < 1e-12
print('calibrated policy writer passed')
PY

echo "== Dendro discovery and progress contracts =="
python -m pytest -q tests/test_dendro_real_checkpoint_support.py

echo "== Existing generic command and Dendro migration regression =="
STATE_ROOT_NAME="$STATE_ROOT_NAME" \
  scripts/validate_two_node_generic_command_dendro.sh

echo "ALL TWO-NODE V1 PARITY CLOSEOUT CHECKS PASSED"
