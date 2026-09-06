# Stage 5E.3 — Real-process destination contention

Stage 5E.3 is the physical analogue of Stage 5C. It preserves the same
controlled four-source contention geometry and production lowest-score
destination arbiter, but replaces the lightweight counter processes with the
actual checkpointable `benchmark-json-medium` workload used in Stage 4A and
Stage 5E.2.

## Geometry

Destination: Ethiopia.

- one real `benchmark-json-medium` process starts resident on Ethiopia;
- four real benchmark challengers start on Boston, California, South Australia,
  and Virginia;
- all five definitions use the exact frozen Stage 4D.1 measured request;
- all four challengers are explicitly evaluated concurrently at the frozen
  Stage 5B/5C trace time and must independently select Ethiopia.

The measured request is approximately 0.9972 CPU cores and 13 MB RAM on a
2-core destination. Therefore:

- resident + one challenger fits;
- resident + two challengers does not fit.

The destination arbiter must admit exactly one of the four competing bids and
reject the other three because of resource contention.

## Physical evidence

Before arbitration, the runner captures a direct process-session witness and
requires all five real benchmark processes to be non-zombie with positive RSS.
This proves the resident pressure and all challenger workloads are physically
executing rather than merely declaring resource demand.

After arbitration and ownership convergence, a second direct process witness
requires:

- the resident still physically live on Ethiopia;
- exactly one migrated challenger physically live on Ethiopia;
- the three rejected challengers still physically live at their source owners;
- five of five real processes live in total.

The same run captures `/health` resource-ledger state from all seven nodes.
Reservations must exactly equal the frozen benchmark request multiplied by the
active owner count on each node. Ethiopia must own two tasks and reserve exactly
two benchmark requests without exceeding capacity.

## PASS criteria

- five real benchmark processes live before arbitration;
- four source daemons independently select migrate-to-Ethiopia;
- four destination bids are observed;
- exactly one bid is accepted/consumed;
- exactly three bids are rejected specifically for scarce resources;
- exactly one real checkpoint/migration completes and no migration fails;
- ownership converges across all seven daemons;
- five real processes live after arbitration;
- exactly two real benchmark processes are simultaneously live on Ethiopia;
- all seven ResourceLedger snapshots match physical ownership and frozen
  Stage 4D.1 reservations;
- no node exceeds configured resource capacity;
- all five tasks stop or complete cleanly during cleanup.

Stage 5E.3 does not compare carbon-aware policies. Stage 5E.4 combines real
heterogeneous workloads, real pressure, autonomous scheduling, queueing, and
policy comparison.
