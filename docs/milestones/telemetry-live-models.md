# Telemetry and Live Models Milestone

## Scope

This milestone replaces configured task, network, and migration-overhead values
with durable live measurements whenever fresh data exists. It preserves safe
configured fallbacks and does not yet adapt alpha, beta, or gamma.

## Implemented components

- Linux process-group CPU and RSS telemetry through procfs.
- Optional Intel RAPL package-power reader with task CPU-share allocation.
- CPU-utilization power model and configured fallback.
- Checkpoint-directory size measurement.
- HTTP peer-latency probes.
- Real rsync throughput feedback into directed graph edges.
- Checkpoint, transfer, restore, activation, and total-downtime calibration.
- Durable telemetry state at `control/telemetry.json`.
- Fresh, stale, and unavailable classifications.
- Automatic fallback to cluster and task configuration.
- Live power use in carbon accounting and scheduling profiles.
- Live bandwidth, latency, checkpoint, and restore values in migration scoring.
- Telemetry provenance in task bids and migration candidate details.

## API

```text
GET /telemetry
GET /telemetry/tasks
GET /telemetry/tasks/{task_id}
GET /telemetry/edges
GET /telemetry/edges/{destination_node_id}
GET /telemetry/calibration
```

`GET /health` also reports telemetry record counts and the durable state path.

## Local installation

Start from the completed auction branch:

```bash
cd ~/Magellan-V2
git status --short
git switch resource-aware-auction
git pull --ff-only
git switch -c telemetry-live-models

git apply --check /path/to/telemetry-live-models.patch
git apply /path/to/telemetry-live-models.patch

source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check magellan tests
python -m compileall -q magellan
pytest -q
```

Expected:

```text
All checks passed!
57 passed
```

Commit:

```bash
git add README.md pyproject.toml magellan config tests scripts docs
git commit -m "Add durable live telemetry models"
git push -u origin telemetry-live-models
```

## GCP deployment

Use three sessions: Boston daemon, Virginia daemon, and a second Boston test
session. On both nodes:

```bash
cd ~/Magellan-V2
git fetch origin

if git show-ref --verify --quiet refs/heads/telemetry-live-models; then
  git switch telemetry-live-models
  git branch --set-upstream-to=origin/telemetry-live-models telemetry-live-models
  git pull --ff-only
else
  git switch -c telemetry-live-models --track origin/telemetry-live-models
fi

source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check magellan tests
python -m compileall -q magellan
pytest -q
```

Both nodes must report the same commit and version `0.8.0`.

Stop previous daemons on both nodes:

```bash
pkill -f 'uvicorn magellan.api.app:app' || true
sleep 2
ss -ltnp | grep ':8040' || true
```

### Boston daemon

```bash
cd ~/Magellan-V2
source .venv/bin/activate
rm -rf "$PWD/runtime-state-telemetry"

export MAGELLAN_NODE_ID=boston
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-telemetry"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-telemetry"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=""
export PYTHONUNBUFFERED=1

python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

### Virginia daemon

Use the same block with:

```bash
export MAGELLAN_NODE_ID=virginia
```

Do not delete the state root during a restart-persistence test.

### Automated validation

From the second Boston session:

```bash
cd ~/Magellan-V2
source .venv/bin/activate

export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2

scripts/validate_two_node_telemetry.sh
```

Expected final line:

```text
ALL TWO-NODE TELEMETRY LIVE-MODEL CHECKS PASSED
```

The script proves:

1. both telemetry services are active;
2. Boston obtains a fresh measured peer latency;
3. a dynamically submitted counter produces CPU, RSS, checkpoint, and power
   telemetry;
4. the counter migrates to Virginia;
5. real rsync throughput becomes Boston-to-Virginia bandwidth telemetry;
6. checkpoint, transfer, restore, activation, and downtime calibration is
   persisted;
7. Virginia begins collecting local process telemetry after activation;
8. the task completes normally; and
9. both nodes have durable telemetry files.

## Manual restart-persistence check

Before restarting Boston:

```bash
curl -fsS "$BOSTON_API/telemetry/edges/virginia" > /tmp/edge-before.json
curl -fsS "$BOSTON_API/telemetry/calibration" > /tmp/cal-before.json
```

Stop only the Boston daemon with `Ctrl+C`, then restart it with the same state
root and without `rm -rf`. Query again:

```bash
curl -fsS "$BOSTON_API/telemetry/edges/virginia" > /tmp/edge-after.json
curl -fsS "$BOSTON_API/telemetry/calibration" > /tmp/cal-after.json

python - <<'PY'
import json
before=json.load(open('/tmp/edge-before.json'))
after=json.load(open('/tmp/edge-after.json'))
assert before['bandwidth_sample_count']==after['bandwidth_sample_count']
assert before['bandwidth_mbps_ema']==after['bandwidth_mbps_ema']
cb=json.load(open('/tmp/cal-before.json'))
ca=json.load(open('/tmp/cal-after.json'))
assert cb==ca
print('DURABLE TELEMETRY RESTART CHECK PASSED')
PY
```

A new latency probe may increment only the latency sample count after restart;
transfer and migration-calibration values must remain intact.

## Manual stale-data fallback check

Stop Virginia, leave Boston running, and wait longer than the development edge
staleness threshold (60 seconds):

```bash
sleep 65
curl -fsS "$BOSTON_API/telemetry/edges/virginia" | python -m json.tool
```

Expected:

```text
latency_freshness: stale
latency_source: configured_fallback
bandwidth_source: configured_fallback   # after its transfer sample also ages out
```

Restart Virginia and wait for the next probe. Latency should return to `fresh`
and `measured_http_rtt`.

## Acceptance criteria

- Ruff and all 57 tests pass on both nodes.
- Task CPU and memory telemetry is fresh.
- Power reports RAPL, procfs model, or configured fallback with confidence.
- Peer latency becomes measured.
- A real transfer produces measured bandwidth.
- Future migration estimates identify the measured bandwidth and calibration
  sources.
- Telemetry survives daemon restart.
- Stale telemetry falls back to configured values.
- The automated validation prints its final pass line.
