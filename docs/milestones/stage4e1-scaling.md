# Stage 4E.1 — 25/50/100-task measured-capacity scaling

## Question

How does Magellan behave as the submitted population grows from 25 to 50 to
100 tasks on the same seven-node measured cluster?

Stage 4B–4D establish carbon efficacy, long-duration dynamics, measured
capacity, utilization sensitivity, and arbiter semantics. Stage 4E.1 isolates
the **population-scaling axis** while retaining the production decision engine
and resource-aware destination auctions.

## Frozen evidence

Stage 4E.1 is rooted through canonical Stage 4D.4 and inherits:

- Stage 4A.1 measured/configured node CPU, memory, GPU and WAN evidence;
- Stage 4A.2 migration/checkpoint calibration;
- Stage 4A.3 p95 CPU/memory resource requests and measured workload power;
- Stage 4A.4 isolated per-node slowdown factors;
- Stage 4A.5 runtime-model approval;
- corrected production scoring from Stage 4B/4C;
- Stage 4D.1 measured resource capacity;
- Stage 4D.2/4D.3 capacity semantics;
- Stage 4D.4 production destination-arbiter policies.

No synthetic task-slot capacity is added.

## Population sizes

The canonical sizes are:

- 25 tasks;
- 50 tasks;
- 100 tasks.

Workload classes rotate deterministically:

1. `benchmark-json-medium`
2. `dendro-r9-t1p0`
3. `llm-distilgpt2`

so each population differs by at most one task per class.

Home regions rotate over the seven configured nodes. Because three workload
classes and seven regions are coprime, the class/location pairing naturally
cycles rather than pinning one workload to one region.

## Arrival process

All three population sizes use the same fixed **3-hour arrival window**, snapped
to the configured scheduler epoch.

Increasing `N` therefore increases offered load without stretching the
submission horizon.

Every task has a deterministic home region. A task begins in a source-side
admission queue and consumes no compute resource until its frozen Stage 4D.1
resource vector fits its home node.

Admission uses a feasibility-preserving FIFO scan: older tasks are considered
first, but an incompatible head task does not force otherwise usable physical
resources to remain idle.

This is intentionally different from pretending all 100 tasks are already
running on seven oversubscribed VMs.

## Work duration

Each task represents **3 Boston-equivalent hours of useful work** by default.

This is intentionally shorter than the 72-hour Stage 4C/4D experiments.
Long-duration behavior is already evaluated there; repeating 72 hours for 100
tasks would primarily multiply identical scheduler epochs rather than answer the
scaling question.

Three hours remains long enough for production migration scoring and measured
migration overheads while making the 25/50/100 sweep operationally tractable.

## Policies

1. `static_resource_queue`
   - resource-aware source admission;
   - task remains on its home node;
   - no migration bids.

2. `magellan_capacity_lowest_score`
   - production `evaluate_task`;
   - one selected destination bid per running task;
   - production `rank_bids(lowest_score)`;
   - measured destination resource ledger.

3. `magellan_capacity_credit_fair`
   - identical scheduler;
   - production `rank_bids(credit_fair)` and persistent destination credits.

Queued tasks do not migrate before admission. Once admitted they are ordinary
running Magellan tasks.

## Metrics

For each `N × policy` case:

- drain time;
- throughput in tasks/hour;
- mean and p95 completion latency;
- mean, p95 and max source queue wait;
- maximum queue depth;
- mean cluster CPU utilization;
- lifecycle carbon;
- monetary cost;
- migrations and tasks migrated;
- bid attempts/accepts/rejections;
- distinct regions visited;
- per-workload completion/queue/carbon metrics.

Static-relative drain, completion, carbon, cost, and queue-wait ratios are
included.

## Visualization trace

`event_trace.csv` is intentionally animation-friendly. It records:

- `submitted`;
- `admitted`;
- `migration_start`;
- `migration_finish`;
- `completed`.

Each record includes timestamp, task, workload, source and destination. The
full bid stream remains in `auction_events.csv` so a globe animation does not
need to render thousands of rejected retries.

## Interpretation boundary

As in Stage 4D.2/4D.3, resource **admission** is measured, but performance
interference between co-located tasks has not been measured. Admitted tasks
therefore retain their isolated Stage 4A.3 power and Stage 4A.4 slowdown
profiles.

Stage 4E.1 is an offline scaling evaluation of the measured scheduler model,
not a claim that 100 physical VMs were simultaneously executed.

## Outputs

- `population.csv`
- `task_outcomes.csv`
- `scaling_summary.csv`
- `per_class_summary.csv`
- `auction_events.csv`
- `migration_events.csv`
- `migration_matrix.csv`
- `occupancy_timeline.csv`
- `event_trace.csv`
- `metadata.json`
- `summary.json`
- `checksums.sha256`
