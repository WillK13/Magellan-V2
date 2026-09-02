# Stage 4E.2 — Control-plane scaling at 25/50/100 tasks

## Question

What is the actual single-process scheduler/arbiter overhead as the evaluated
task batch grows from 25 to 50 to 100 tasks?

Stage 4E.1 establishes measured-capacity workload scaling. Stage 4E.2 measures
the control-plane code itself so a 100-task scalability claim is supported by
wall-clock scheduler evidence rather than only simulated task throughput.

## Timed production path

Each task invokes the production `evaluate_task` implementation with the same:

- Stage 4A workload calibration;
- Stage 4A.4 node slowdown;
- Stage 4D.1 resource request;
- seven-node cluster graph;
- production scoring policy;
- summer carbon timestamp used by Stage 4E.1.

After scoring, every task contributes its best migration candidate as one bid.
Bids are grouped by the destination selected by production scoring and processed
through:

1. production `rank_bids(lowest_score)`;
2. measured `ResourceLedger` compatibility;
3. measured resource reservation for feasible ranked bids.

Thus the benchmark measures both decision-engine and destination-arbiter work.

## Measurements

For each N in 25, 50 and 100:

- cold first-pass epoch wall and process CPU time;
- median and p95 steady-state decision wall time;
- median decision time per task;
- decision tasks/second;
- median and p95 auction wall time;
- auction microseconds per bid;
- auction bids/second;
- median and p95 combined control-plane epoch time;
- process CPU time for the combined epoch;
- tasks/second for the combined epoch;
- separate peak incremental Python allocation via `tracemalloc`.

Seven timed repetitions are used by default after two warmups.

## Cache methodology

The cold first pass is retained as a separate metric.

Warmup passes populate the existing process-local `ReplayCarbonStore` cache.
The headline median/p95 numbers therefore represent steady-state repeated
scheduler operation rather than repeatedly charging dataset lookup and forecast
construction as if every epoch came from a fresh process.

Cache hit/entry deltas are written to the summary for auditability.

## Adaptive policy state

Every repetition receives a fresh `AdaptivePolicyService` store so one timed
sample cannot inherit adaptive state from another. Store construction occurs
outside the timed epoch, while the normal adaptive operations inside
`evaluate_task` remain included.

## Memory methodology

`tracemalloc` is deliberately run in a separate epoch because allocation
instrumentation changes execution latency. The reported peak is incremental
Python-tracked allocation, not whole-VM RSS.

## Scope boundary

This is a single-process offline control-plane microbenchmark. It does not
include:

- FastAPI/HTTP transport;
- cross-region network RPC;
- CRIU/checkpoint I/O;
- task execution;
- GCP VM scheduling noise outside the Python process.

Those belong to the system and migration experiments, not the scheduler
algorithm benchmark.

## Pass criteria

PASS validates benchmark coverage and internal consistency only. There is no
latency, throughput, scaling-slope or memory threshold. Performance is a result,
not a criterion the experiment can tune itself to satisfy.

## Outputs

- `control_plane_summary.csv`
- `latency_samples.csv`
- `metadata.json`
- `summary.json`
- `checksums.sha256`
