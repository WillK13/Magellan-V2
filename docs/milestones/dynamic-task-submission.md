# Dynamic task submission milestone

## Goal

Allow a user to submit an immutable workload definition and create durable task runs through any Magellan peer without listing those tasks in `MAGELLAN_TASK_FILES` or restarting the daemon.

## Branch

```text
dynamic-task-submission
```

This branch is based on `durable-distributed-reconciliation`.

## Implemented data model

### Definition

A definition is a reusable immutable workload template. It contains:

- logical definition ID;
- revision;
- SHA-256 digest;
- workload/scoring profile;
- runtime command contract;
- artifact manifest;
- origin peer and creation time.

Submitting an identical payload is idempotent and returns the existing revision. Submitting a changed payload under the same definition ID creates the next revision.

### Run

A task run is one execution of a definition revision. Each run has:

- generated `run-<UUID>` ID;
- definition ID, revision, and digest;
- initial owner;
- client idempotency key;
- immutable request digest;
- optional labels.

A completed definition can therefore be launched again as a separate run without reusing checkpoint, accounting, or output state.

## Durable storage

```text
runtime-state-dynamic-submission/
├── control/
│   └── task_catalog.json
└── tasks/
    └── run-<uuid>/
        ├── state.json
        ├── checkpoint/
        ├── runtime/
        └── output/
```

The catalog uses atomic temporary-file replacement. On daemon startup, catalog runs are materialized into the persistent task registry before runtime reconciliation begins.

## API

```text
POST /task-definitions
GET  /task-definitions
GET  /task-definitions/{definition_id}?revision=N

POST /task-runs
GET  /task-runs
GET  /task-runs/{run_id}
POST /task-runs/{run_id}/start
POST /task-runs/{run_id}/stop

GET  /catalog/snapshot
```

Existing `/tasks/{run_id}/...` lifecycle endpoints remain valid for pause, resume, migration, output retrieval, and other operations.

## Decentralized replication

The existing anti-entropy service now pulls both catalog snapshots and ownership snapshots. It merges immutable definitions first, materializes new runs into the local registry, and then applies generation-fenced ownership updates. This order ensures Virginia knows the task definition before it receives ownership changes for the run.

## Task bidding direction

Tasks bid for destination capacity. If many tasks want France, each task submits a destination-specific bid to France, and France's local arbiter selects the best feasible task bids. Every bid carries structured task context for future resource-aware ranking. See `docs/design/task-bidding-market.md`.

## Local installation

```bash
cd ~/Magellan-V2
git switch durable-distributed-reconciliation
git pull --ff-only
git switch -c dynamic-task-submission

git apply --check /path/to/dynamic-task-submission.patch
git apply /path/to/dynamic-task-submission.patch

source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m compileall -q magellan
pytest -q
```

Expected:

```text
38 passed
```

Commit:

```bash
git add README.md pyproject.toml magellan config tests scripts docs
git commit -m "Add decentralized dynamic task submission"
git push -u origin dynamic-task-submission
```

## Two-node GCP validation

### Environment on both nodes

Use an empty static task-file list to prove runtime submission is sufficient:

```bash
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-dynamic-submission"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-dynamic-submission"
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

From a second Boston session:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2

scripts/validate_two_node_dynamic_submission.sh
```

Expected final line:

```text
ALL TWO-NODE DYNAMIC TASK SUBMISSION CHECKS PASSED
```

The script proves:

1. Both daemons start with no static task definitions.
2. Boston accepts an immutable definition.
3. Repeating the same definition is idempotent.
4. Boston creates and auto-starts one durable run.
5. Retrying the run request returns the same run ID.
6. Virginia learns the definition and run through anti-entropy.
7. The dynamically created task pauses and resumes.
8. The task bids for Virginia capacity and migrates.
9. The consumed bid identifies the bidder as the task and carries task context.
10. The task completes on Virginia and publishes final output.
11. A second independent run can be created from the same definition.
12. Durable catalog files exist on both peers.

## Manual restart persistence check

After the script passes, stop both daemons with `Ctrl+C`. Do not delete the state root. Restart both using the same environment, including `MAGELLAN_TASK_FILES=""`.

Then run:

```bash
curl -fsS http://127.0.0.1:8040/task-definitions | python -m json.tool
curl -fsS http://127.0.0.1:8040/task-runs | python -m json.tool
curl -fsS http://10.162.0.2:8040/task-definitions | python -m json.tool
curl -fsS http://10.162.0.2:8040/task-runs | python -m json.tool
```

Both peers must still contain the submitted definition and both task runs. The first run remains completed and owned by Virginia; the second remains stopped and initially owned by Boston.

## Deliberate non-goals

This milestone does not yet implement:

- multi-resource feasibility enforcement;
- task cancellation and retry lineage;
- user authentication or quotas;
- globally serialized simultaneous submissions to disconnected peers;
- adaptive weights;
- SLURM or Dendro adapters.

The next recommended branch is `resource-aware-auction`, because dynamic task runs now carry the metadata required for tasks to compete intelligently for destinations such as France.
