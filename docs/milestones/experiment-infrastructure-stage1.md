# Experiment Infrastructure Stage 1

Goal: prove that a single autonomous seven-node workload can be reconstructed from
structured, durable evidence before any paper-result campaign begins.

Stage 1 adds an append-only per-daemon experiment event journal. Scheduler decision
events contain the complete ranked action list, selected action, raw and normalized
metrics, policy metadata, task profile, checkpoint information, and trace time.
Successful migrations additionally record measured checkpoint, transfer, restore,
activation, downtime, and byte counts.

The recorded-counter driver captures the event sequence boundary before submission,
runs one dynamic counter task, polls ownership/task/telemetry/policy state, collects
all task-specific evidence from all seven nodes, and writes an immutable bundle under
`experiments/runs/<experiment-id>/`.

Each bundle contains:

- `manifest.json`: Git/config/policy/workload/dataset identity and hashes.
- `raw/<node>.json`: per-node events, bids, migrations, ownership, telemetry and state.
- `raw/events.jsonl`: merged structured experiment events.
- `raw/observations.jsonl`: time-series task/telemetry/policy observations.
- `decisions.csv`: one row per autonomous scheduler decision.
- `decision_candidates.csv`: every ranked CONTINUE/PAUSE/MIGRATE candidate.
- `migrations.csv`: measured migration outcomes and timings.
- `ownership.csv`: observed ownership/generation/status transitions.
- `task_results.csv`: final runtime/carbon/cost accounting.
- `summary.json`: high-level experiment result.
- `checksums.sha256`: integrity hashes for the entire bundle.

Stage 1 is validation infrastructure only. It does not introduce paper baselines,
oracles, LLM/Dendro campaigns, sensitivity sweeps, or result figures.
