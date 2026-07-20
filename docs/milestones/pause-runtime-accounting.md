# Milestone: pause runtime and task accounting

Branch: `pause-runtime-accounting`

Parent milestone: `task-lifecycle-recovery`

## Objective

Close the gap between Magellan's three modeled actions and the actions the decentralized daemon actually enforces. This milestone makes `PAUSE` a real runtime state and adds persistent task accounting for progress, time, cost, carbon, and migration overhead.

## Delivered behavior

### Real pause and resume

A pause uses process-group signals:

```text
SIGSTOP -> PAUSED -> SIGCONT -> RUNNING
```

The paused process keeps the same PID and in-memory state. The task registry persists:

```text
status
pid
paused_at_utc
resume_at_utc
resume_wall_at_utc
last_pause_at_utc
pause_reason
pause_count
```

`resume_at_utc` is in Magellan's evaluation-clock domain. `resume_wall_at_utc` is separately persisted so a daemon restart cannot strand the task when an accelerated trace clock restarts from its configured trace origin.

The `PauseService` scans paused tasks independently of the scheduler epoch and resumes them when the wall deadline is due.

### Scheduler action execution

The scheduler now performs all three outcomes:

```text
CONTINUE -> no runtime change
PAUSE    -> freeze process and schedule resume
MIGRATE  -> bid, reserve, prefetch, transfer, activate
```

A configurable `min_pause_gap_seconds` prevents an immediately resumed task from being paused again at the next epoch.

### Manual API

```text
POST /tasks/{task_id}/pause?idle_seconds=<evaluation-seconds>
POST /tasks/{task_id}/resume
```

The manual pause uses evaluation seconds. Under the development trace scale of 60, a 300-second evaluation pause lasts approximately 5 wall-clock seconds.

### Standard progress contract

A workload may publish `runtime/progress.json`:

```json
{
  "format_version": 1,
  "task_id": "counter-pause-001",
  "completed_units": 40,
  "total_units": 240,
  "updated_at_utc": "2026-07-19T12:00:00Z",
  "node_id": "boston",
  "details": {
    "unit": "counter-value"
  }
}
```

The counter and LLM workloads now emit this record atomically. The daemon estimates throughput with an exponential moving average and derives remaining evaluation time.

### Persistent ledger

Each `TaskRuntimeState` now includes:

```text
estimated_remaining_seconds
accumulated_runtime_seconds
accumulated_paused_seconds
accumulated_migration_seconds
accumulated_compute_cost_usd
accumulated_transfer_cost_usd
accumulated_cost_usd
accumulated_compute_carbon_grams
accumulated_transfer_carbon_grams
accumulated_carbon_grams
progress_completed_units
progress_total_units
progress_fraction
progress_rate_units_per_second
progress_updated_at_utc
last_accounted_at_utc
```

Compute accounting uses the owner node's configured price, PUE, task power, and carbon trace. In trace mode, wall elapsed time is converted into evaluation time using `trace_seconds_per_real_second`.

Pause idle time contributes to paused duration but follows the original Magellan assumption that idle task time is not charged as active compute. Actual SIGSTOP/SIGCONT operation latency is negligible compared with the configured prospective pause model.

### Migration handoff

Before migration, Boston settles running accounting. Successful checkpoint and missing-artifact transfer adds:

```text
actual transferred bytes
source egress cost
network energy/carbon estimate
measured migration downtime
```

An accounting snapshot is included in the activation request and ownership broadcast. Virginia therefore continues from Boston's accumulated budget and carbon state instead of resetting them to zero.

### Cost cap

`PersistentTaskRegistry.scoring_profile()` now overrides the static task JSON with live:

```text
estimated_remaining_seconds
accumulated_cost_usd
last_pause_at
last_migration_at
```

Migration pruning therefore uses the task's actual accumulated compute and transfer cost.

## Files added

```text
magellan/runtime/accounting.py
magellan/runtime/pause.py
magellan/runtime/progress.py
config/tasks/dev-counter-pause.json
config/tasks/dev-counter-budget.json
config/tasks/dev-counter-accounting-migrate.json
tests/test_pause_runtime.py
tests/test_runtime_accounting.py
tests/test_accounting_handoff.py
tests/test_pause_scoring.py
tests/test_scheduler_pause_action.py
tests/test_cost_cap_accounting.py
docs/design/adaptive-weights-v2.md
scripts/validate_two_node_pause_accounting.sh
```

## Local installation and tests

```bash
cd ~/Magellan-V2
git switch task-lifecycle-recovery
git switch -c pause-runtime-accounting

git apply --check /path/to/pause-runtime-accounting.patch
git apply /path/to/pause-runtime-accounting.patch

source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m compileall -q magellan
pytest -q
```

Expected:

```text
23 passed
```

## Two-node deployment

Run on both Boston and Virginia:

```bash
cd ~/Magellan-V2
git fetch origin
git switch pause-runtime-accounting
git pull --ff-only
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Use the following task files on both daemons:

```bash
export MAGELLAN_TASK_FILES=config/tasks/dev-counter-pause.json,config/tasks/dev-counter-budget.json,config/tasks/dev-counter-accounting-migrate.json
```

Use an isolated state root:

```bash
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-pause-accounting"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-pause-accounting"
```

Start the Boston daemon with `MAGELLAN_NODE_ID=boston` and the Virginia daemon with `MAGELLAN_NODE_ID=virginia`, using the existing development cluster and policy files.

Then, from Boston:

```bash
scripts/validate_two_node_pause_accounting.sh
```

## Manual daemon-restart pause check

1. Start `counter-pause-001` on Boston.
2. Pause it for 600 evaluation seconds:

```bash
curl -fsS -X POST \
  'http://127.0.0.1:8040/tasks/counter-pause-001/pause?idle_seconds=600' \
  | python -m json.tool
```

With the development scale of 60, this is a 10-second wall pause.

3. Stop only the Boston daemon with `Ctrl+C` while the workload remains SIGSTOP'd.
4. Restart the daemon with the same state root within the 10-second window.
5. Inspect the task:

```bash
curl -fsS http://127.0.0.1:8040/tasks | python -m json.tool
```

Expected: it loads as `paused` with the same PID, then automatically changes to `running` when `resume_wall_at_utc` arrives.

## Adaptive weights

This milestone deliberately retains fixed configured weights. It supplies the reliable accumulated cost, carbon, and remaining-time inputs required by the V1 adaptive design. See `docs/design/adaptive-weights-v2.md`.

## Next milestone

The recommended next branch is:

```text
durable-distributed-reconciliation
```

It should persist bids and migration transactions, add migration outcome queries, repair missed ownership broadcasts, and reconcile daemon restarts without manual intervention.
