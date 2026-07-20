# Magellan V2

**A decentralized, carbon-aware scheduler for long-running stateful workloads.**

Magellan V2 runs the same daemon at every compute site. There is no central controller. Each node manages its locally owned tasks, evaluates **continue**, **pause**, and **migrate**, and coordinates directly with peers.

Tasks bid for capacity at destination nodes. The destination ranks competing task bids, reserves resources, receives missing artifacts, and activates the transferred checkpoint. Ownership, accounting, telemetry, policy state, and recovery metadata survive migrations and daemon restarts.

## Features

- Decentralized scheduling and ownership
- Time, carbon, and monetary-cost optimization
- Hard cost caps, deadlines, priorities, and adaptive weights
- Short-horizon carbon forecasting and pause-duration search
- Resource-aware destination auctions
- Artifact prefetch before checkpointing
- Durable checkpoint transfer, restart, and reconciliation
- CPU, memory, power, progress, latency, and bandwidth telemetry
- Python-module, generic-command, and Dendro runtime adapters
- Application checkpoints for LLM and Dendro workloads

```text
        Boston                                  Virginia
  ┌──────────────────┐                    ┌──────────────────┐
  │ Magellan daemon  │◄──── peer API ────►│ Magellan daemon  │
  │ local scheduler  │                    │ local scheduler  │
  │ task ownership   │                    │ task ownership   │
  │ telemetry        │                    │ telemetry        │
  │ bid arbiter      │                    │ bid arbiter      │
  │ checkpoint store │◄── SSH / rsync ───►│ checkpoint store │
  └──────────────────┘                    └──────────────────┘
```

## Install

Requirements: Linux for deployment, Python 3.11, Git, OpenSSH, `rsync`, passwordless SSH between nodes, and TCP access to port `8040`.

```bash
git clone <repository-url> ~/Magellan-V2
cd ~/Magellan-V2

git switch main
git pull --ff-only

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For LLM training:

```bash
python -m pip install -e '.[llm]'
python -m pip install torch
```

Verify:

```bash
ruff check magellan tests scripts
python -m compileall -q magellan scripts
python -m pytest -q
```

## Configure and start two nodes

Use `config/cluster.dev.json` for Boston/Virginia or copy it for another deployment. Update node IDs, IPs, carbon datasets, resource capacity, prices, PUE, and network properties. Every node must use the same cluster file.

Common environment on both nodes:

```bash
cd ~/Magellan-V2
source .venv/bin/activate

export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER="$USER"
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

Check the cluster:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040

curl -fsS "$BOSTON_API/health" | python -m json.tool
curl -fsS "$BOSTON_API/peers" | python -m json.tool
curl -fsS "$BOSTON_API/capabilities" | python -m json.tool
```

## Time, carbon, and cost policy

Magellan minimizes a normalized weighted score:

```text
score = time_weight × makespan
      + carbon_weight × carbon
      + cost_weight × monetary_cost
```

Configure baseline weights in `config/policy.dev.json`:

```json
{
  "weights": {
    "time": 0.25,
    "carbon": 0.50,
    "cost": 0.25
  }
}
```

Weights are normalized internally.

| Goal | Time | Carbon | Cost |
|---|---:|---:|---:|
| Default carbon-aware | 0.25 | 0.50 | 0.25 |
| Balanced | 1 | 1 | 1 |
| Fastest completion | 0.70 | 0.20 | 0.10 |
| Lowest carbon | 0.15 | 0.70 | 0.15 |
| Lowest cost | 0.15 | 0.15 | 0.70 |

Adaptive weights are enabled by default:

```json
{
  "adaptive": {
    "enabled": true,
    "multiplier_bound_fraction": 0.25
  }
}
```

Set `enabled` to `false` for fixed weights.

Per-task controls:

```json
{
  "cost_cap_usd": 10.0,
  "deadline_at_utc": "2026-08-01T12:00:00Z",
  "priority": 50
}
```

- Cost caps are hard constraints applied before scoring.
- Deadlines increase time urgency.
- Priority is used by the `priority_deadline` auction.

### Default run

Use `config/policy.dev.json` unchanged. Magellan automatically chooses continue, pause, or migration.

### Local throughput baseline

Use a one-node cluster, disable adaptive weights, use `time=1`, `carbon=0`, `cost=0`, and set `candidate_idle_seconds` to `[0]`.

## Destination auction strategies

Set `auction.strategy` in the policy or override it at startup:

```bash
export MAGELLAN_AUCTION_STRATEGY=credit_fair
```

| Strategy | Destination ranking behavior |
|---|---|
| `lowest_score` | Lowest Magellan migration score |
| `shortest_remaining` | Least remaining work |
| `longest_remaining` | Most remaining work |
| `credit_fair` | Rejected tasks gain future credit |
| `highest_regret` | Worst fallback or highest opportunity loss |
| `priority_deadline` | Highest priority and deadline urgency |
| `resource_efficiency` | Most migration value per resource share |

**Tasks bid for destination capacity; destinations do not bid for tasks.**

## Run the LLM workload

The LLM runtime checkpoints model weights, tokenizer files, optimizer state, completed steps, and PyTorch RNG state.

Prepare the dataset:

```bash
mkdir -p ~/magellan-artifacts/llm-001-dataset
cat > ~/magellan-artifacts/llm-001-dataset/train.txt <<'DATASET'
Magellan migrates stateful workloads between carbon-aware computing regions.
Long-running jobs can continue, pause, or migrate based on time, carbon, and cost.
DATASET
```

Update `source_directory` in `config/tasks/dev-llm-v2.json`. Start both daemons with:

```bash
export MAGELLAN_TASK_FILES=config/tasks/dev-llm-v2.json
```

Start and inspect the task from Boston:

```bash
curl -fsS -X POST "$BOSTON_API/tasks/llm-001/start" | python -m json.tool
curl -fsS "$BOSTON_API/telemetry/tasks/llm-001" | python -m json.tool
curl -fsS "$BOSTON_API/policy/tasks/llm-001" | python -m json.tool
```

Request a migration manually:

```bash
curl -fsS -X POST "$BOSTON_API/tasks/llm-001/migrate/virginia" \
  | python -m json.tool
```

Stop it on its current owner:

```bash
curl -fsS -X POST "$VIRGINIA_API/tasks/llm-001/stop" | python -m json.tool
```

## Run Dendro

### Validation workload

Submit the included multi-process Dendro-compatible definition:

```bash
curl -fsS \
  -H 'Content-Type: application/json' \
  --data-binary @config/submissions/dev-dendro-definition.json \
  "$BOSTON_API/task-definitions" \
  | python -m json.tool
```

Create and auto-start a run:

```bash
RUN_ID="$(
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d '{
      "definition_id": "dendro-bssn-validation",
      "initial_owner_node_id": "boston",
      "idempotency_key": "dendro-validation-001",
      "auto_start": true
    }' \
    "$BOSTON_API/task-runs" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["run"]["run_id"])'
)"

echo "$RUN_ID"
```

Inspect and migrate:

```bash
curl -fsS "$BOSTON_API/telemetry/tasks/$RUN_ID" | python -m json.tool
curl -fsS -X POST "$BOSTON_API/tasks/$RUN_ID/migrate/virginia" \
  | python -m json.tool
curl -fsS "$VIRGINIA_API/task-runs/$RUN_ID" | python -m json.tool
```

### Real Dendro-GR

```bash
cp config/submissions/dendro-bssn-template.json \
   config/submissions/dendro-bssn-local.json
```

Set the real solver path, parameter file, MPI world size, checkpoint patterns, restart arguments, and progress-log expression. Validate before submission:

```bash
python scripts/validate_real_dendro_definition.py \
  config/submissions/dendro-bssn-local.json
```

Submit it through `/task-definitions` and `/task-runs` using definition ID `dendro-bssn-real`.

Magellan only migrates Dendro after discovering a complete, stable checkpoint and confirming destination compatibility.

## Manual controls

```bash
curl -X POST "$API/tasks/$TASK_ID/pause?idle_seconds=900"
curl -X POST "$API/tasks/$TASK_ID/resume"
curl -X POST "$API/tasks/$TASK_ID/migrate/$DESTINATION_NODE_ID"
curl -X POST "$API/tasks/$TASK_ID/stop"
```

## Useful endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Node and service health |
| `GET /peers` | Peer reachability |
| `GET /capabilities` | Hardware and runtime capabilities |
| `GET /tasks` | Task summaries |
| `GET /task-definitions` | Durable definitions |
| `GET /task-runs` | Durable task runs |
| `GET /auction/status` | Capacity, strategy, and fairness credits |
| `GET /bids` | Bid history and explanations |
| `GET /telemetry/tasks/{task_id}` | CPU, memory, power, and progress |
| `GET /telemetry/edges/{node_id}` | Latency and bandwidth |
| `GET /policy/tasks/{task_id}` | Effective weights and decisions |
| `GET /carbon/forecast/{node_id}` | Carbon forecast and confidence |
| `GET /migrations` | Migration records |
| `GET /tasks/{task_id}/outputs` | Final output manifest |

## Validation

```bash
scripts/validate_two_node_v1_parity_closeout.sh
```

Other subsystem validators are available under `scripts/`.

## Current boundaries

The current release uses application-level checkpoints. Future extensions include JAX/Orbax, SLURM, CRIU, GPU-state migration, and live Electricity Maps ingestion.

Detailed design documentation is under `docs/design/` and `docs/milestones/`.
