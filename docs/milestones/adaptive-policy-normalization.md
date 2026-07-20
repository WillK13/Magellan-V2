# Milestone: adaptive-policy-normalization

## Scope

This milestone implements offline baseline calibration, bounded per-task runtime adaptation, rolling normalization, explainable decision records, durable policy state, and migration/reconciliation handoff.

It deliberately does not add new runtime adapters or expand the deployment beyond the existing peer architecture.

## Branch

```bash
git switch telemetry-live-models
git pull --ff-only
git switch -c adaptive-policy-normalization
```

## Local validation

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check magellan tests
python -m compileall -q magellan
pytest -q
```

Expected:

```text
All checks passed!
67 passed
```

## Configuration

Development policy:

```json
{
  "adaptive": {
    "enabled": true,
    "multiplier_bound_fraction": 0.25,
    "rolling_window_epochs": 24,
    "decision_history_limit": 50,
    "confidence_floor": 0.25
  }
}
```

Production uses a longer rolling window and history.

## GCP daemon environment

Use the same environment on both peers except `MAGELLAN_NODE_ID`:

```bash
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-adaptive-policy"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-adaptive-policy"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=""
export PYTHONUNBUFFERED=1
```

Boston:

```bash
export MAGELLAN_NODE_ID=boston
python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

Virginia:

```bash
export MAGELLAN_NODE_ID=virginia
python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

Do not retain `MAGELLAN_TEST_FORCE_ACTIVATION_RESPONSE_LOSS` from an earlier reconciliation test.

## Automated two-node test

From a second Boston shell:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2

scripts/validate_two_node_adaptive_policy.sh
```

Expected final line:

```text
ALL TWO-NODE ADAPTIVE POLICY CHECKS PASSED
```

The script verifies:

1. Both daemons expose adaptive-policy health metadata.
2. Offline calibration enforces hard constraints.
3. An urgent, nearly budget-exhausted task raises time and cost priority.
4. Multipliers remain inside ±25% and effective weights sum to one.
5. Cost-cap pruning occurs before adaptive scoring.
6. Rolling normalization records real candidate ranges.
7. A second task carries policy state from Boston to Virginia.
8. Ownership anti-entropy contains the policy snapshot.
9. Durable policy files exist on both peers.

## Manual restart persistence

Before restart:

```bash
curl -fsS "$VIRGINIA_API/policy/tasks/<run-id>" \
  > /tmp/policy-before-restart.json
```

Stop and restart the Virginia daemon without deleting `runtime-state-adaptive-policy`. Then:

```bash
curl -fsS "$VIRGINIA_API/policy/tasks/<run-id>" \
  > /tmp/policy-after-restart.json

python - <<'PY'
import json
before=json.load(open('/tmp/policy-before-restart.json'))
after=json.load(open('/tmp/policy-after-restart.json'))
assert after['decision_count'] >= before['decision_count']
assert after['baseline_weights'] == before['baseline_weights']
assert after['normalization']['time']['samples']
print('ADAPTIVE POLICY RESTART PERSISTENCE PASSED')
PY
```

## Acceptance criteria

- Ruff passes.
- All 67 tests pass.
- Effective weights sum to one.
- Multipliers are bounded.
- Cost-cap pruning cannot be overridden.
- Rolling normalization spans epochs.
- Policy state survives daemon restart.
- Policy state follows task ownership.
- Automated GCP test prints its final pass line.
