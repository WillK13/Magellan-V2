# Stage 5B — Real multi-origin decentralized scheduling

## Question

Do multiple **actual source daemons** independently execute Magellan's production
scheduler and send real peer-to-peer bids, with ownership converging across the
seven-node deployment?

Stage 5A proves identical deployment and 42/42 connectivity. Stage 5B is the
first experiment whose scheduling decisions are produced concurrently by
multiple real Magellan daemons rather than by a Boston-hosted replay.

## Sources

Four geographically distinct sources originate one task each:

- Boston
- California
- South Australia
- Virginia

The lightweight counter workload is checkpointable and deliberately requests
only 0.1 CPU / 64 MB. Stage 5B is a **decentralization/control-path test**, not a
capacity stress test; measured-capacity contention is isolated in Stage 5C.

## Controlled trigger

Runs are submitted with `scheduler_mode=operator_only`. This prevents the
15-minute background loop from racing the measurement.

Stage 5B adds `POST /tasks/{task_id}/evaluate`, an operator trigger that calls
the same production `SchedulerService._evaluate_task_locked` path used by the
background scheduler. The trigger does **not** supply a destination or action.
The owning daemon independently performs:

1. checkpoint validation;
2. telemetry/resource enrichment;
3. production continue/pause/migrate scoring;
4. destination selection;
5. one peer bid if migration is selected;
6. destination-side production arbitration;
7. real migration when the bid is accepted.

Boston sends the four evaluate-now HTTP triggers concurrently. It coordinates
timing only; it does not calculate or override a decision.

## Trace time

All four evaluations use the same controlled summer timestamp
`2024-08-20T12:00:00Z`. This removes small trace-clock skew caused by sequential
systemd startup while retaining the production carbon datasets and scoring.

## Evidence

The bundle records:

- source/task/trigger identity;
- structured `scheduler_decision` events from every daemon;
- all new destination-side bid records;
- migration completed/failed events;
- all seven ownership snapshots for all four tasks;
- final owner/generation/status;
- raw per-node event/bid/ownership evidence.

## PASS

PASS requires:

- all four real source triggers succeed;
- each task has its scheduler decision recorded **only by its true origin daemon**;
- at least two distinct source daemons emit real bids;
- every bid is stored by the actual destination daemon;
- no migration failure is recorded;
- ownership converges for all four tasks across all seven nodes;
- the running daemon SHA matches the source Stage 5A bundle.

PASS deliberately does not require a preferred destination, number of accepted
bids, or carbon result.
