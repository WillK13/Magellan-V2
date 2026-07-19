# Milestone: durable-distributed-reconciliation

This milestone starts from `pause-runtime-accounting` and makes the decentralized control plane recoverable across daemon restarts, missed ownership broadcasts, and lost migration activation responses.

## Guarantees added

1. **Bids and reservations are durable.** Every mutation is atomically written to `STATE_ROOT/control/bids.json`. Pending, accepted, activating, consumed, cancelled, rejected, and expired records survive daemon restarts.
2. **Every migration has a durable transaction record.** Source and destination independently store `STATE_ROOT/control/migrations/<migration-id>.json`.
3. **Activation outcomes are queryable.** `GET /migrations/{migration_id}` returns the destination's durable status.
4. **Lost activation responses are resolved safely.** The source queries the destination journal. It commits remote ownership only after observing `activated`; it restarts locally only after the destination confirms `rolled_back` or repeatedly confirms the migration was never recorded.
5. **Interrupted destination activation is deterministic after restart.** A committed local owner is finalized and its reservation consumed. An incomplete activation is rolled back from the saved task state and checkpoint backup.
6. **Ownership is repaired by anti-entropy.** Every peer periodically fetches `/ownership/snapshot` from all other peers and applies higher-generation ownership records.
7. **Equal-generation owner conflicts are rejected.** This avoids replacing one owner with a conflicting peer claim at the same fencing generation.

## Durable layout

```text
runtime-state-reconciliation/
├── control/
│   ├── bids.json
│   └── migrations/
│       └── <migration-id>.json
└── tasks/
    └── <task-id>/
        ├── state.json
        ├── checkpoint/
        └── ...
```

## Migration states

```text
preparing
  → transferring
  → activating
      ├── activated
      ├── rolled_back
      └── uncertain
```

`uncertain` is not treated as failure. The source remains stopped and the reconciliation loop continues querying the destination until it can make a fenced decision.

## API additions

```text
GET /migrations
GET /migrations/{migration_id}
GET /ownership/snapshot
```

`GET /health` additionally reports:

```text
migration_record_count
last_reconciliation_at_utc
last_reconciliation_updates
```

## Configuration

Development policy:

```json
"reconciliation": {
  "enabled": true,
  "scan_interval_seconds": 1,
  "activation_resolution_timeout_seconds": 8,
  "activation_resolution_poll_seconds": 0.5
}
```

Production policy uses a slower anti-entropy scan and a longer immediate activation-resolution window.

## Important files

New:

```text
magellan/migration/journal.py
magellan/reconciliation/__init__.py
magellan/reconciliation/models.py
magellan/reconciliation/client.py
magellan/reconciliation/service.py
config/tasks/dev-counter-reconciliation.json
tests/test_durable_bid_store.py
tests/test_migration_journal.py
tests/test_migration_restart_reconciliation.py
tests/test_ownership_reconciliation.py
tests/test_unknown_activation_resolution.py
scripts/validate_two_node_reconciliation.sh
```

Modified:

```text
magellan/bidding/store.py
magellan/migration/models.py
magellan/migration/client.py
magellan/migration/service.py
magellan/state/persistent_registry.py
magellan/config/policy_models.py
magellan/daemon/context.py
magellan/api/app.py
config/policy.dev.json
config/policy.prod.json
pyproject.toml
README.md
```

## Local installation and tests

```bash
cd ~/Magellan-V2
git switch pause-runtime-accounting
git pull --ff-only
git switch -c durable-distributed-reconciliation

git apply --check /path/to/durable-distributed-reconciliation.patch
git apply /path/to/durable-distributed-reconciliation.patch

source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m compileall -q magellan
pytest -q
```

Expected:

```text
32 passed
```

Commit:

```bash
git add README.md pyproject.toml magellan config tests scripts docs
git commit -m "Add durable migration and ownership reconciliation"
git push -u origin durable-distributed-reconciliation
```

# Two-node GCP validation

Use three terminals: Boston daemon, Virginia daemon, and a second Boston shell.

## 1. Pull the same commit on both nodes

```bash
cd ~/Magellan-V2
git fetch origin
git switch durable-distributed-reconciliation
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Expected on both nodes:

```text
32 passed
```

Confirm `git rev-parse HEAD` is identical.

## 2. Start Boston

Use a clean test root. The fault-injection variable intentionally discards the successful activation response after Virginia has committed; the source must recover by querying Virginia's durable journal.

```bash
cd ~/Magellan-V2
source .venv/bin/activate
rm -rf "$PWD/runtime-state-reconciliation"

export MAGELLAN_NODE_ID=boston
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-reconciliation"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-reconciliation"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=config/tasks/dev-counter-reconciliation.json
export MAGELLAN_TEST_FORCE_ACTIVATION_RESPONSE_LOSS=1
export PYTHONUNBUFFERED=1

python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

## 3. Start Virginia

```bash
cd ~/Magellan-V2
source .venv/bin/activate
rm -rf "$PWD/runtime-state-reconciliation"

export MAGELLAN_NODE_ID=virginia
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-reconciliation"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-reconciliation"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=config/tasks/dev-counter-reconciliation.json
export PYTHONUNBUFFERED=1

python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

## 4. Run the automatic validation from Boston

```bash
cd ~/Magellan-V2
source .venv/bin/activate

export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2

scripts/validate_two_node_reconciliation.sh
```

Expected final line:

```text
ALL TWO-NODE DURABLE RECONCILIATION CHECKS PASSED
```

The source log should contain:

```text
[migration-resolved] task=counter-reconciliation-001 outcome=activated
```

Both source and destination journals should report `activated`, and Virginia's bid should report `consumed`.

# Manual restart validation

## A. Durable accepted reservation

Start both daemons with the same state root, then submit a synthetic bid to Virginia:

```bash
python - <<'PY' >/tmp/durable-bid.json
import json, uuid
from datetime import datetime, timezone
print(json.dumps({
  "bid_id": "restart-" + str(uuid.uuid4()),
  "epoch_id": "manual-restart",
  "task_id": "reservation-restart-probe",
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
    "score": 0
  },
  "submitted_at_utc": datetime.now(timezone.utc).isoformat()
}))
PY

BID_ID="$(python -c 'import json; print(json.load(open("/tmp/durable-bid.json"))["bid_id"])')"
curl -fsS -X POST -H 'Content-Type: application/json' \
  --data-binary @/tmp/durable-bid.json \
  http://10.162.0.2:8040/bids | python -m json.tool
sleep 4
curl -fsS "http://10.162.0.2:8040/bids/${BID_ID}" | python -m json.tool
```

It should be `accepted`. Stop only the Virginia daemon with `Ctrl+C`, restart it with the same `MAGELLAN_STATE_ROOT` without deleting the directory, then query the same bid again. It must still exist. Depending on elapsed time it will be `accepted` or `expired`, never missing.

## B. Repair a deliberately stale Boston ownership record

First complete a Boston→Virginia migration and leave the task running on Virginia. Stop only the Boston daemon. Back up and deliberately stale Boston's local task state:

```bash
cd ~/Magellan-V2
cp runtime-state-reconciliation/tasks/counter-reconciliation-001/state.json \
   /tmp/counter-reconciliation-state-good.json

python - <<'PY'
import json
from pathlib import Path
path=Path('runtime-state-reconciliation/tasks/counter-reconciliation-001/state.json')
state=json.loads(path.read_text())
state['owner_node_id']='boston'
state['generation']=0
state['status']='stopped'
state['pid']=None
state['last_migration_id']=None
path.write_text(json.dumps(state, indent=2)+'\n')
PY
```

Restart Boston with the same state root. Within a few seconds:

```bash
curl -fsS http://127.0.0.1:8040/tasks | python -m json.tool
```

Boston must converge back to:

```text
owner_node_id: virginia
generation: 1
status: remote
```

The Boston log should show `[reconcile-applied]`.

## C. Terminal migration state survives daemon restart

After a successful migration, stop and restart Virginia without deleting the state root:

```bash
curl -fsS http://10.162.0.2:8040/migrations | python -m json.tool
curl -fsS http://10.162.0.2:8040/bids | python -m json.tool
```

The destination migration remains `activated` and its reservation remains `consumed`.

## Safety behavior

If the destination cannot be reached after an activation response is lost, the source remains stopped and the source journal remains `uncertain`. It does not restart locally and risk duplicate execution. The reconciliation loop continues attempting to determine the destination outcome.

# Relationship to adaptive weights

This branch intentionally does not change alpha, beta, gamma. Durable ownership and accounting are prerequisites for adaptive policy state: a peer must know which node owns the authoritative budget, progress, and normalization history before it can safely update effective weights. The later `adaptive-policy-normalization` milestone will build on these reconciliation guarantees.
