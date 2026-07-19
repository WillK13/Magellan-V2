# Resource-aware task auction milestone

## Branch

```text
resource-aware-auction
```

Based on `dynamic-task-submission`.

## Scope

This milestone converts the destination arbiter from a task-slot-only lowest-score sorter into a configurable, resource-aware task auction.

Implemented ranking strategies:

```text
lowest_score
shortest_remaining
longest_remaining
credit_fair
highest_regret
priority_deadline
resource_efficiency
```

Tasks bid for destination capacity. Destinations never bid for tasks.

## Resource model

Each node may configure:

```json
{
  "capacity": 2,
  "resources": {
    "cpu_cores": 2,
    "memory_mb": 4096,
    "gpu_count": 0,
    "accelerator_types": []
  }
}
```

`capacity` limits the number of simultaneous owned or reserved tasks. The nested resource limits prevent CPU, memory, GPU, and accelerator overcommit.

A null resource limit disables enforcement for that resource, which preserves compatibility with older cluster configurations.

## Persistent credits and explainability

`control/bids.json` now stores both bid records and destination-local task credits. Every decided bid records:

- auction strategy;
- rank within its window;
- credit before and after the decision;
- whether the request is individually resource-feasible;
- candidate score, `Tc`, opportunity loss, priority, deadline slack, dominant resource share, and resource efficiency;
- requested cores, memory, and GPUs.

Use:

```text
GET /auction/status
```

for current strategy, task slots, available resources, and credits.

## Local installation

```bash
cd ~/Magellan-V2
git switch dynamic-task-submission
git pull --ff-only
git switch -c resource-aware-auction

git apply --check /path/to/resource-aware-auction.patch
git apply /path/to/resource-aware-auction.patch

source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m compileall -q magellan
pytest -q
```

Expected:

```text
47 passed
```

Commit:

```bash
git add README.md pyproject.toml magellan config tests scripts docs
git commit -m "Add resource-aware task auctions"
git push -u origin resource-aware-auction
```

## Two-node GCP validation

Run both daemons with no static tasks and an isolated state root:

```bash
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-resource-auction"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-resource-auction"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=""
export MAGELLAN_AUCTION_STRATEGY=credit_fair
export PYTHONUNBUFFERED=1
```

Set `MAGELLAN_NODE_ID=boston` or `virginia` and start Uvicorn on port 8040.

From Boston:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2
scripts/validate_two_node_resource_auction.sh
```

Expected final line:

```text
ALL TWO-NODE RESOURCE-AWARE AUCTION CHECKS PASSED
```

The script proves:

1. The daemon reports `credit_fair` and explicit resource capacity.
2. Two tasks compete for one Virginia slot.
3. The lower-score task wins the first window.
4. The rejected task receives durable credit.
5. In the next window, its credit lets it beat a lower-score task.
6. An oversized CPU bid is rejected as resource-infeasible and receives no credit.
7. A dynamically submitted task generates a normal task-to-Virginia bid.
8. That bid carries fallback/opportunity-loss and resource metrics.
9. The accepted reservation is consumed during real migration.

## Strategy experiments

Change only:

```bash
export MAGELLAN_AUCTION_STRATEGY=<strategy>
```

then restart the destination daemon with a clean experimental state root. Supported values are listed above. This makes comparative bidding experiments reproducible without editing source code.

## Deliberate non-goals

This milestone does not yet implement:

- bandwidth sharing between simultaneous transfers;
- preemption of tasks already running at a destination;
- user/account quotas;
- adaptive alpha/beta/gamma weights;
- live resource telemetry replacing configured capacities;
- SLURM or GPU runtime adapters.

The next recommended branch is `telemetry-live-models`, followed by `adaptive-policy-normalization` after its inputs are trustworthy.
