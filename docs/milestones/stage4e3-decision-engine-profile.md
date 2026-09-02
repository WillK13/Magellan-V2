# Stage 4E.3 — Decision-engine hotspot attribution

## Motivation

Canonical Stage 4E.2 shows that Magellan completes a 100-task control-plane
epoch well inside the configured scheduler interval, but decision cost per task
increases with population size:

- 25 tasks: ~25 ms/task;
- 50 tasks: ~35 ms/task;
- 100 tasks: ~59 ms/task.

The auction remains only a few milliseconds, so Stage 4E.3 asks **why the
decision path grows superlinearly** before any optimization is attempted.

## Method

Stage 4E.3 does not alter the scheduler.

For 25, 50 and 100 tasks it reconstructs the exact Stage 4E.2 benchmark
population and production path:

1. production `evaluate_task`;
2. best migration-candidate bid construction;
3. production `rank_bids(lowest_score)`;
4. measured `ResourceLedger` admission.

An unprofiled epoch first warms the process-local `ReplayCarbonStore`. A fresh
adaptive-policy state directory is then used for one `cProfile`-instrumented
epoch.

The canonical latency values remain those from Stage 4E.2 because profiler
instrumentation adds overhead. Stage 4E.3 is attribution evidence, not a
replacement latency benchmark.

## Pre-profile hypothesis

Inspection of the production adaptive-state path suggests one plausible
superlinear source:

- `AdaptivePolicyService.prepare()` calls `store.put()`;
- `AdaptivePolicyService.record_decision()` calls `store.put()` again;
- `AdaptivePolicyStore.put()` calls `_persist()`;
- `_persist()` serializes **every task state currently in the store**, writes a
  new JSON file, flushes, `fsync`s, and atomically replaces the previous file.

Within one N-task epoch this can create an increasing amount of serialization
and durable-file work as the state dictionary grows.

This is only a hypothesis. Stage 4E.3 PASS does not require the store to be the
hotspot.

## Outputs

### `profile_summary.csv`

For each scale:

- profiled epoch wall time;
- Stage 4E.2 canonical unprofiled median;
- profiler overhead ratio;
- calls/cumulative time for:
  - `evaluate_task`;
  - `build_raw_actions`;
  - continue/pause/migrate estimators;
  - `forecast_or_average`;
  - adaptive `prepare`;
  - adaptive `record_decision`;
  - `AdaptivePolicyStore.put`;
  - `AdaptivePolicyStore._persist`;
  - `fsync`;
- persistent-store share of profiled wall time;
- dominant self-time category.

### `function_profile.csv`

Top cumulative-time profiler functions per scale. Cumulative values overlap
through the call stack and are intended to locate expensive paths.

### `category_profile.csv`

Non-overlapping profiler **self time** grouped into:

- scheduler scoring;
- continue/pause/migration estimators;
- carbon forecast/store;
- adaptive policy/store;
- bidding/auction;
- Pydantic;
- pandas;
- JSON serialization;
- filesystem I/O;
- other Magellan/runtime code.

## Interpretation

If the adaptive durable-state path dominates and its cost grows rapidly with N,
the next step is an implementation optimization followed by a clean Stage 4E.2
rerun.

If it does not dominate, the function/category profiles identify the actual
source and no speculative optimization is made.

## Pass criteria

PASS validates profile coverage and provenance only. No function must dominate,
and no optimization opportunity is required.
