# Experiment Infrastructure Stage 2: Baselines + Offline Oracle

Goal: establish strong, reproducible comparison policies before running expensive
LLM/Dendro or multi-job GCP campaigns.

Stage 2 is offline analysis infrastructure. It does not modify the seven-node daemon,
auction, migration, or scheduling implementation. It reuses Magellan's v1.2 scoring
and physical migration equations where applicable and produces checksummed comparison
bundles under `experiments/comparisons/`.

## Policies

- **Boston Static**: execute the entire workload in Boston.
- **France Static**: execute the entire workload in France.
- **Best Static**: clairvoyantly evaluate the full workload interval on every node,
  choose one fixed node using the configured time/carbon/cost weights, and assume free
  initial placement. This is intentionally stronger than an online scheduler.
- **Best at Dispatch**: at submission time, use only Magellan's causal carbon forecast
  and configured objective weights to choose one of the seven nodes, then never move.
  Initial placement is free because the job has not started yet.
- **Temporal Only**: run Magellan's causal scoring loop with migration candidates
  disabled. Continue and pause remain available.
- **Magellan Causal Replay**: replay the v1.2 scoring implementation using the seven
  carbon traces and configured network fallbacks. This is a model-validation replay,
  not a substitute for measured seven-node Magellan runs because it does not contain
  live edge/power telemetry or destination contention.
- **Clairvoyant Oracle**: discretized shortest-path lower bound that sees the full
  future carbon trace. Unlike Best Static, it starts at the same submission node as
  Magellan and pays Magellan's modeled migration overhead. Its WAIT action is
  deliberately optimistic (zero task-attributed carbon/cost), so it is labeled a
  clairvoyant reference rather than a deployable scheduler.

## Fairness boundaries

The policies deliberately separate two questions reviewers raised:

1. Could a workload simply be placed in a clean region before it starts? Best Static
   and Best at Dispatch answer that question with free initial placement.
2. Once a workload has started at the submission site, how much does online migration
   help and how far is greedy scheduling from clairvoyant planning? Magellan Causal
   Replay and the Oracle both begin at the configured start node.

Stage 2 assumes static artifacts are pre-staged for causal replay/oracle comparisons.
The benefit and cost of prefetching are evaluated separately in the planned prefetch
ablation so static-data transfer does not silently advantage one scheduling policy.

## Output

`run_baseline_comparison.py` writes:

- `manifest.json`: commit/config/policy/dataset hashes and workload identity.
- `metadata.json`: policy semantics, objective reference scales, and objective values.
- `results.csv`: one comparable row per policy.
- `results.json`: structured outcomes.
- `trajectories/<policy>.json`: every simulated action and owner transition.
- `checksums.sha256`: bundle integrity hashes.

The Stage-2 output is for framework/model validation first. It is not automatically a
paper result. Final NSDI figures should pair these references with recorded real-system
runs from Stage 1 and later measured migration/network calibration stages.
