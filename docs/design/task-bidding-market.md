# Magellan V2 task-to-location auction semantics

Magellan's auction direction is intentionally **task to destination**.

1. A running task evaluates `CONTINUE`, `PAUSE`, and one `MIGRATE` candidate per feasible destination.
2. If migration is selected, the task's current owner submits a task bid to that destination.
3. The destination-local arbiter gathers all task bids received during its bid window.
4. Tasks compete for the destination's scarce capacity. The destination never bids for tasks.
5. Accepted bids become expiring resource reservations. Rejected tasks remain at their current owners and can bid again in a later epoch.

For example, if Boston, California, and Virginia tasks all prefer France, each task submits a bid to France. France ranks those task bids and accepts only the best feasible set for its available capacity.

## Current ranking

The current slot-based arbiter ranks by the task's normalized Magellan scheduling score, with receive time and bid ID as deterministic tie-breakers. This preserves the existing time/carbon/cost objective.

Every bid now also carries a `TaskBidContext`:

- workload type;
- priority;
- deadline;
- estimated remaining work;
- checkpoint and missing static-data bytes;
- accumulated and capped cost;
- CPU, memory, GPU, and accelerator requirements.

The current arbiter records but does not yet enforce all of those fields. The later `resource-aware-auction` milestone will use them to determine feasibility and ranking, including task urgency, carbon benefit per reserved resource, checkpoint size, transfer contention, deadlines, and fairness.

## Invariants

- A bid always names exactly one task and one desired destination.
- `bidder_type` is always `task`.
- A destination accepts task bids; it never advertises a payment or sends a bid to a task.
- The task remains owned by its source until activation commits.
- An accepted task bid is only a lease, not ownership.
- Generation-fenced ownership changes only after destination activation succeeds.
