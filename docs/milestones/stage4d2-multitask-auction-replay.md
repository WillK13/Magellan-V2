# Stage 4D.2 — Measured-Capacity Multi-Task Auction Replay

## Goal

Stage 4D.2 is the first synchronized multi-task contention experiment. It asks:

> When every node starts resource-full under measured workload demand, how does
> production Magellan behave as resources free over time and multiple tasks bid
> for the same destination?

The experiment deliberately does not add a synthetic task-slot limit. Admission is
based on the Stage 4D.1 CPU/memory/GPU resource vectors.

## Frozen inputs

Stage 4D.2 consumes:

- the canonical corrected Stage 4B bundle and its Stage 4A.1–4A.5 provenance;
- the canonical corrected Stage 4C bundle for one deterministic arrival per season
  and the 72-hour Boston-equivalent useful-work target;
- the Stage 4D.1 node capacities and p95 workload requests;
- Stage 4A.4 node slowdown factors;
- Stage 4A.1/4A.2 affine migration calibration;
- lifecycle carbon traces and configured prices;
- the corrected production `evaluate_task` policy.

## Initial population

Each seasonal scenario starts 11 simultaneous tasks:

- 4 `benchmark-json-medium`;
- 3 `dendro-r9-t1p0`;
- 4 `llm-distilgpt2`.

Every node receives exactly one maximal packing proven feasible by Stage 4D.1. The
seven packing templates are:

1. 2 benchmark tasks;
2. 1 benchmark + 1 LLM;
3. 1 Dendro;
4. 2 LLM;
5. 1 Dendro;
6. 1 benchmark + 1 LLM;
7. 1 Dendro.

The sequence rotates over the configured node order once per season. This keeps the
global task mix fixed while avoiding a permanent association between one region and
one workload class. Because every initial node packing is maximal, the cluster starts
admission-full for the three measured workloads even though small unusable CPU
fragments can remain.

## Scheduler and auction semantics

At each configured scheduler epoch:

1. every runnable task evaluates the real production `evaluate_task` policy;
2. a task that selects migration emits one bid only, to the selected destination,
   matching `SchedulerService`;
3. all bids to the same destination in that synchronized trace epoch are ranked by
   the production `rank_bids` implementation;
4. destination admission uses the production `ResourceLedger` and the Stage 4D.1
   capacity vector;
5. rejected tasks remain at their current owner and continue running until the next
   scheduler epoch;
6. capacity-based rejection accrues destination-local credit using the configured
   `AuctionPolicy`;
7. accepted migration pays the frozen Stage 4A.1/4A.2 transfer/checkpoint/restore
   overhead and then continues on the destination;
8. Stage 4A.4 slowdown determines useful-work progress and therefore when resources
   become free.

Outbound tasks are not treated as free destination capacity during the same auction
round. This conservative rule avoids implicit preemption or atomic task swaps that the
production arbiter does not implement.

## Policies

The canonical replay reports four policies:

- `static_initial_layout`: no scheduling or migration;
- `magellan_unlimited_reference`: each task runs an independent causal Magellan
  replay without capacity constraints, showing the placement pressure that would
  exist with unlimited resources;
- `magellan_capacity_lowest_score`: measured capacity with the base destination
  auction;
- `magellan_capacity_credit_fair`: identical measured capacity with persistent
  rejection credit.

Movement, multi-hop paths, and contention are observed results, not pass criteria.

## Co-location boundary

Stage 4D.1 measures whether workload resource requests fit concurrently, but Stage
4A.4 measured isolated regional slowdown. Therefore Stage 4D.2 assumes an admitted
task retains its isolated Stage 4A.4 slowdown and Stage 4A.3 power estimate while
co-located with another admitted task.

This is an explicit model boundary, not a claim that co-location interference was
measured. If Stage 4D.2 becomes a headline paper result, a later co-location
validation campaign should measure the four Stage 4D.1 maximal packing types on final
hardware and quantify any runtime/power interference.

## Outputs

The bundle contains:

- `initial_layout.csv`;
- `task_outcomes.csv`;
- `scenario_outcomes.csv`;
- `policy_summary.csv`;
- `auction_events.csv`;
- `migration_events.csv`;
- `migration_matrix.csv`;
- `occupancy_timeline.csv`;
- `metadata.json`, `summary.json`, and checksums.

The validator requires all tasks to complete and verifies that the measured-capacity
policies never exceed CPU, memory, or GPU capacity. It does not require any minimum
number of migrations or rejections.

## Canonical command

```bash
python scripts/run_stage4d2_multitask_auction.py \
  --stage4d1-bundle "$MAGELLAN_STAGE4D1_CANONICAL" \
  --stage4c-bundle "$MAGELLAN_STAGE4C_CANONICAL"
```

Then validate:

```bash
D42=$(find experiments/measurements -maxdepth 1 -type d -name 'stage4d2-*' \
  -print | sort | tail -1)
python scripts/validate_stage4d2_multitask_auction.py "$D42"
```

Expected marker:

```text
STAGE_4D2_MULTITASK_AUCTION_BUNDLE_PASS
```
