# Milestone: Task lifecycle, output publication, reservation leases, and recovery

Branch name: `task-lifecycle-recovery`

Base: completed `artifact-prefetch` milestone.

## Scope and invariants

This milestone keeps Magellan V2 fully decentralized. Every peer still runs the same FastAPI daemon and no new controller is introduced.

The lifecycle invariants are:

1. An accepted bid is an expiring capacity reservation, not permanent ownership.
2. Only a matching, unexpired reservation may activate an incoming migration.
3. Destination activation is transactional: checkpoint replacement, artifact staging, ownership claim, process startup, and reservation consumption either all succeed or roll back.
4. A process exit is successful only when the workload atomically writes its configured completion marker.
5. Final outputs are immutable files described by a SHA-256 manifest and served only by the completing owner.
6. A crashed local owner restarts from its last valid checkpoint with bounded exponential backoff.
7. An unknown remote-activation outcome fails closed: the source stays stopped rather than risking split-brain execution.

## File changes

### State and task contract

- `magellan/state/task_models.py`
  - Adds `RECOVERING` and `COMPLETED` states.
  - Adds completion/output paths to `LocalProcessSpec`.
  - Adds persistent exit, failure, recovery, completion, and output-manifest metadata.
  - Validates all runtime-relative paths.
- `magellan/state/persistent_registry.py`
  - Adds completion/output path helpers.
  - Adds persistent completion and recovery transitions.
  - Excludes completed tasks from capacity use.
  - Applies terminal completion broadcasts using generation fencing.

### Completion and final output handling

- `magellan/runtime/completion.py` (new)
  - Validates workload completion markers.
  - Hashes final outputs and atomically publishes `final-output/manifest.json`.
  - Safely resolves output-download paths.
- `magellan/runtime/local_process.py`
  - Renders `{completion_file}` and `{output_directory}`.
  - Classifies natural exit as `COMPLETED` only after marker validation.
  - Classifies unmarked exit as `FAILED`.
  - Returns reconcile events for completion broadcast.
- `magellan/workloads/counter.py`
  - Adds finite-run support and writes `output/result.json` followed by the completion marker.
- `magellan/workloads/llm_train.py`
  - Writes `output/training-summary.json` and a completion marker only after reaching `--max-steps` naturally.
- `magellan/api/app.py`
  - Adds output-manifest and output-file endpoints.

### Reservation expiry and activation fencing

- `magellan/bidding/models.py`
  - Adds `ACTIVATING`, `CONSUMED`, `CANCELLED`, and `EXPIRED` states and lease timestamps.
- `magellan/bidding/store.py`
  - Adds lease acceptance, renewal, expiry, activation claim, consumption, and cancellation.
- `magellan/bidding/arbiter.py`
  - Subtracts active reservations from available capacity and expires old leases.
- `magellan/bidding/client.py`
  - Adds reservation renewal and cancellation RPCs.
- `magellan/config/models.py`
  - Adds lease TTL and renewal interval with validation.
- `magellan/migration/models.py`
  - Carries `bid_id` into activation and terminal lifecycle fields in ownership broadcasts.

### Failure recovery and transactional migration

- `magellan/runtime/recovery.py` (new)
  - Bounded exponential-backoff recovery from a valid checkpoint.
- `magellan/migration/service.py`
  - Renews the destination reservation throughout prefetch/transfer/activation.
  - Requires the destination to claim the matching reservation.
  - Rolls back checkpoint, state, and reservation after destination startup failure.
  - Restarts the source after explicit migration failure.
  - Fails closed after an unknown activation outcome.
- `magellan/migration/client.py`
  - Distinguishes an unknown activation result from an explicit rejection.
- `magellan/daemon/context.py`
  - Wires the completion manager, recovery service, shared broadcaster, and lease-aware bid store.
- `magellan/daemon/scheduler_service.py`
  - Broadcasts completion and passes accepted bid IDs into migration.
  - Adds an operator-triggered migration path that still uses scoring, bidding, leases, and the normal migration transaction.

### Configuration and validation assets

- `config/cluster.dev.json`: 15-second development lease and 5-second renewal.
- `config/cluster.gcp.json`: complete seven-node schema plus production lease settings.
- `config/policy.dev.json` and `config/policy.prod.json`: bounded recovery policy.
- `config/tasks/dev-counter-completion.json`: finite migration/completion test.
- `config/tasks/dev-counter-recovery.json`: crash/restart test.
- `config/tasks/dev-counter.json` and `config/tasks/dev-llm-v2.json`: completion/output contract.
- `scripts/validate_two_node_lifecycle.sh`: end-to-end Boston/Virginia validation.

## Local tests

```bash
cd ~/Magellan-V2
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m compileall -q magellan
pytest -q
```

Expected result for this milestone snapshot:

```text
17 passed
```

## Deploy to Boston and Virginia

Run on the development machine containing the new branch:

```bash
cd ~/Magellan-V2
git switch artifact-prefetch
git pull --ff-only
git switch -c task-lifecycle-recovery
# Apply the milestone patch or copy the supplied milestone tree.
pytest -q
git add magellan config tests scripts docs
git commit -m "Add task lifecycle leases outputs and recovery"
git push -u origin task-lifecycle-recovery
```

On Boston:

```bash
gcloud compute ssh instance-20251125-025803 \
  --zone us-east1-c

cd ~/Magellan-V2
git fetch origin
git switch task-lifecycle-recovery
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

On Virginia:

```bash
gcloud compute ssh instance-20251125-033020 \
  --zone northamerica-northeast1-c

cd ~/Magellan-V2
git fetch origin
git switch task-lifecycle-recovery
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Verify passwordless internal SSH in both directions because checkpoint and artifact transport use rsync over SSH:

```bash
ssh -o BatchMode=yes WILL@10.162.0.2 true   # from Boston
ssh -o BatchMode=yes WILL@10.142.0.2 true   # from Virginia
```

## Start both peer daemons

Use a clean validation state root so older task ownership cannot affect the test. Run these commands on both nodes, changing only `MAGELLAN_NODE_ID`.

Boston:

```bash
cd ~/Magellan-V2
source .venv/bin/activate
rm -rf runtime-state-lifecycle-dev

export MAGELLAN_NODE_ID=boston
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-lifecycle-dev"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-lifecycle-dev"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=config/tasks/dev-counter-completion.json,config/tasks/dev-counter-recovery.json

uvicorn magellan.api.app:app \
  --host 0.0.0.0 --port 8040
```

Virginia:

```bash
cd ~/Magellan-V2
source .venv/bin/activate
rm -rf runtime-state-lifecycle-dev

export MAGELLAN_NODE_ID=virginia
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-lifecycle-dev"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-lifecycle-dev"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=config/tasks/dev-counter-completion.json,config/tasks/dev-counter-recovery.json

uvicorn magellan.api.app:app \
  --host 0.0.0.0 --port 8040
```

Do not delete a production state root. The deletion above is only for the isolated development validation root.

## Two-node validation

From a machine with HTTP access to both internal IPs and SSH access to Boston:

```bash
cd ~/Magellan-V2
BOSTON_SSH=WILL@10.142.0.2 \
  scripts/validate_two_node_lifecycle.sh
```

The script validates all of the following:

1. Both peers are healthy.
2. `counter-complete-001` starts in Boston.
3. The operator migration endpoint builds the Virginia candidate, submits a real bid, receives an expiring reservation, prefetched artifacts, transfers the checkpoint, and activates Virginia.
4. The task naturally completes in Virginia at value 80.
5. Virginia publishes and serves a hashed final-output manifest and `result.json`.
6. `counter-recovery-001` is killed with SIGKILL in Boston and automatically restarts from its checkpoint with a new PID.
7. A separate accepted Virginia reservation is left unused and becomes `expired` after its lease TTL.

Expected final line:

```text
ALL TWO-NODE LIFECYCLE CHECKS PASSED
```

## Manual inspection commands

```bash
curl -s http://10.142.0.2:8040/health | python -m json.tool
curl -s http://10.162.0.2:8040/health | python -m json.tool
curl -s http://10.142.0.2:8040/tasks | python -m json.tool
curl -s http://10.162.0.2:8040/tasks | python -m json.tool
curl -s http://10.162.0.2:8040/bids | python -m json.tool
curl -s http://10.162.0.2:8040/tasks/counter-complete-001/outputs \
  | python -m json.tool
curl -s http://10.162.0.2:8040/tasks/counter-complete-001/outputs/result.json \
  | python -m json.tool
```

## Acceptance criteria

The milestone is complete when:

- all 17 local tests pass on both VMs;
- the validation script ends successfully;
- Boston reports the completed task as remote and owned by Virginia;
- Virginia reports the task as completed with no PID;
- the output manifest digest and byte count are present in persistent state;
- the crash test shows `recovery_attempts >= 1` and a new PID;
- the unused probe bid changes from `accepted` to `expired`;
- no peer ever reports two running owners for the same task generation.
