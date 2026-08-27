# Stage 4C — 72-Hour Dynamic Carbon Crossover

## Goal

Stage 4B establishes the representative annual policy comparison. Stage 4C asks a narrower systems question: **when a stateful job remains active long enough for regional carbon conditions to change, does the production causal Magellan controller revisit placement rather than behaving like a one-time dispatcher?**

The experiment never forces migration. A valid Stage 4C bundle may contain zero, one, or many migrations. The scientific result is the behavior produced by the frozen policy under selected real carbon-crossover windows.

## Why 72 hours

Every workload is scaled to exactly **72 hours of Boston-static useful work**. This matches the SC26 evaluation rationale: the window is long enough to capture diurnal carbon cycles and inter-regional variation, while remaining short enough that long-term averaging does not erase transient volatility.

Only useful-work duration is scaled. Stage-4A measurements remain frozen:

- Stage 4A.1 affine WAN transfer calibration,
- Stage 4A.2 workload checkpoint/restore/migration overhead,
- Stage 4A.3 workload power,
- Stage 4A.4 Boston runtime and per-node slowdown factors,
- Stage 4A.5 approval of the single-node slowdown runtime model.

Checkpoint size, power, PUE, regional prices, WAN transfer cost, migration energy, and pause/migration policy thresholds are not multiplied by the 72-hour scaling factor.

## Trace-only crossover selection

Stage 4C is a stress test, not an annual average. Representativeness is already supplied by Stage 4B.

The runner first screens the same 24 deterministic arrivals used by Stage 4B:

- day 5 at 00:00 UTC, and
- day 20 at 12:00 UTC,
- for every month of 2024.

Selection is performed **before any Magellan replay** and depends only on the frozen lifecycle-carbon traces, PUE, and the configured minimum migration gap.

For each 72-hour candidate window, the runner samples carbon hourly and identifies the scheduler-carbon leader: the node minimizing

`lifecycle carbon intensity × PUE`.

Leadership intervals shorter than the configured minimum migration gap are not considered sustained opportunities. Within each season, the canonical campaign chooses exactly one arrival using the following deterministic ranking:

1. most sustained scheduler-carbon leadership transitions,
2. most distinct sustained leaders,
3. most raw hourly leadership changes,
4. earliest UTC arrival as the tie-breaker.

The four selected arrivals (winter, spring, summer, fall) are then replayed for all three frozen workload classes, yielding **12 canonical scenarios**.

A winter window with no crossover is valid and useful: it checks that Magellan does not invent traversal when the traces provide no sustained reason to move.

## Policies

Stage 4C replays only:

- `boston_static`, and
- `magellan_causal`.

The five-policy Boston/best-static/GAIA/oracle comparison remains frozen in Stage 4B. Repeating those baselines here would increase replay time without answering the dynamic-path question.

## Dynamic diagnostics

For each Magellan scenario the bundle records:

- complete owner path,
- migration count and timestamps,
- pause count,
- decision count,
- active compute residence time at each node,
- hourly scheduler-carbon leadership,
- hourly realized-work carbon leadership (`intensity × PUE × slowdown`),
- sustained leadership opportunity windows,
- whether each opportunity was exploited or ignored,
- source/destination carbon intensity for every migration,
- predicted migrate-vs-continue carbon and score differences from the production decision,
- measured migration time/carbon/cost,
- and an offline clairvoyant diagnostic comparing “stay at source” against “migrate once, then stay at destination” for the remaining work.

The clairvoyant diagnostic is analysis only. It never changes the causal Magellan decision.

## Pass criteria

`STAGE_4C_DYNAMIC_CROSSOVER_PASS` means only that the experiment is structurally complete and scientifically auditable:

- all four deterministic seasonal windows are selected from the 24 candidates,
- all three workload classes are run at each selected arrival,
- Boston-static and Magellan complete for all 12 scenarios,
- Boston-static is exactly 72 hours per scenario,
- leadership and migration diagnostics are complete,
- and bundle checksums are valid.

**Multiple migration is intentionally not a pass criterion.** If the result is still migrate-once, that result must be preserved and explained rather than tuning the scheduler until it traverses.
