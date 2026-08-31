# Stage 4D.3 — Measured-utilization capacity sweep

## Question

How does Magellan's benefit degrade as measured cluster utilization increases?

Stage 4D.2 establishes the saturated stress-test endpoint: all seven nodes begin in
Stage 4D.1 maximal resource packings. Stage 4D.3 turns that endpoint into a
utilization curve without introducing synthetic task slots or synthetic resource
requirements.

## Frozen evidence

Stage 4D.3 inherits the exact Stage 4D.2 provenance:

- Stage 4A.1 measured/configured node CPU, memory, GPU, and WAN evidence.
- Stage 4A.2 migration/checkpoint calibration.
- Stage 4A.3 p95 workload CPU/memory requests and power.
- Stage 4A.4 isolated per-node slowdown factors.
- Stage 4A.5 runtime-model validation.
- corrected Stage 4B production scoring.
- Stage 4C 72-hour Boston-equivalent useful-work target.
- Stage 4D.1 evidence-backed resource vectors.
- Stage 4D.2 maximal 4 benchmark / 3 Dendro / 4 DistilGPT2 population.

## Load levels

The first three levels preserve equal workload-class counts:

| Load | Workload population | Approx. measured cluster CPU |
|---|---|---:|
| `u25` | 1 benchmark + 1 Dendro + 1 LLM | 25% |
| `u50` | 2 benchmark + 2 Dendro + 2 LLM | 50% |
| `u75` | 3 benchmark + 3 Dendro + 3 LLM | 76% |
| `umax` | 4 benchmark + 3 Dendro + 4 LLM | 88% |

The percentages are outcomes of the frozen Stage 4D.1 p95 requests divided by
the measured 14-core cluster capacity; they are not synthetic capacity knobs.

Subsets are nested (`u25 ⊂ u50 ⊂ u75 ⊂ umax`) and are chosen to maximize
distinct initial-node coverage before deterministic task-id tie breaking.

## Policies

The sweep isolates the capacity effect with:

1. `static_initial_layout`
2. `magellan_unlimited_reference`
3. `magellan_capacity_lowest_score`

Fairness strategies are intentionally reserved for Stage 4D.4.

## Runtime optimization and exact reuse

A full replay of every policy/load point would repeat deterministic work already
frozen by Stage 4D.2.

Therefore:

- static task outcomes are exact task-level subsets of Stage 4D.2;
- unlimited task outcomes are exact task-level subsets of Stage 4D.2 because
  independent unlimited tasks do not interact;
- the `umax` measured-capacity `lowest_score` result is copied exactly from
  Stage 4D.2;
- only `u25`, `u50`, and `u75` measured-capacity cases are newly replayed.

This changes no scheduling decision and avoids repeating the expensive maximal
stress test.

## Canonical season

The default canonical run uses the frozen Stage 4D.2 **summer** window. In
Stage 4D.2 that window already exhibited capacity-driven migration to Ethiopia,
South Australia, and France, making it a useful representative utilization
sweep while keeping wall-clock cost bounded.

`--season all` repeats the same methodology for all four frozen seasonal
windows if a full seasonal replication is later required.

## Outputs

- `load_cases.csv`
- `initial_layout.csv`
- `task_outcomes.csv`
- `scenario_outcomes.csv`
- `utilization_summary.csv`
- `auction_events.csv`
- `migration_events.csv`
- `migration_matrix.csv`
- `occupancy_timeline.csv`
- `metadata.json`
- `summary.json`
- `checksums.sha256`

The primary paper-facing result is `utilization_summary.csv`: carbon, makespan,
cost, migration activity, and rejection rate as a function of measured initial
CPU utilization.

## Interpretation boundary

Stage 4D.3 preserves the Stage 4D.2 assumption that admitted co-located tasks
retain their isolated Stage 4A.3 power and Stage 4A.4 slowdown. The experiment
tests measured resource admission and scheduler contention, not unmeasured
co-location performance interference.
