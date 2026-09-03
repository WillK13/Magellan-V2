# Stage 5D — Real seven-node migration ring

## Question

Can Magellan's decentralized checkpoint-transfer-restore mechanism move a
stateful workload correctly between arbitrary peers, with every deployed node
acting as both a migration source and a migration destination?

Stages 5B and 5C already test autonomous scheduler decisions and real
destination-side contention. Stage 5D deliberately controls the destination
sequence so it can test the migration mechanism across the full deployed
geography without manipulating carbon traces to force a desired route.

## Route

One checkpointable counter task traverses:

```
Boston
  -> California
  -> South Australia
  -> Nepal
  -> Ethiopia
  -> France
  -> Virginia
  -> Boston
```

This gives seven real migrations. Every GCP node acts exactly once as source
and exactly once as destination.

## What is controlled

The experiment chooses the next destination with the existing operator
migration endpoint.

It does **not** bypass migration machinery. Every hop still executes:

1. source ownership/status checks;
2. destination compatibility validation;
3. checkpoint validation;
4. production migration scoring for the selected peer;
5. a real bid sent to that destination;
6. destination `BidArbiter` and `ResourceLedger` admission;
7. source process stop and checkpoint validation;
8. real rsync checkpoint transfer over the measured peer path;
9. destination activation and process restart;
10. source and destination durable migration journal updates;
11. ownership broadcast to the other peers.

Stage 5D therefore tests mechanism portability, not carbon-policy route choice.

## PASS criteria

PASS requires:

- 7/7 successful migrations;
- all seven nodes act once as source;
- all seven nodes act once as destination;
- generation advances exactly 0 -> 1 -> ... -> 7;
- the destination process is running with a PID after every hop;
- every source migration journal is `source/activated`;
- every destination migration journal is `destination/activated`;
- each migration has a real accepted/consumed bid;
- every hop records positive downtime;
- standardized task progress never decreases;
- ownership converges across all seven peers after every hop;
- final owner is Boston at generation 7.

No downtime threshold is a PASS criterion.

## Outputs

- `hops.csv`
- `ownership_per_hop.csv`
- `final_ownership.csv`
- `migration_events.jsonl`
- `migration_journals.jsonl`
- `initial_state.json`
- `final_state.json`
- `metadata.json`
- `summary.json`
- `checksums.sha256`


## Durable progress evidence

Stage 5D grades application-state continuity at the migration boundary rather
than from the scheduler registry's periodically refreshed progress field. For
every hop, after the source process has stopped, the runner reads the counter's
durable checkpoint, progress file, and final `stopped value` from that node's
effective `MAGELLAN_STATE_ROOT`. It then reads the destination process log and
requires its latest `resumed value` to equal the source checkpoint exactly.
The destination checkpoint and progress file may be greater because the task is
already running again.

Registry `progress_completed_units` remains in `hops.csv` as diagnostic
accounting telemetry (`registry_progress_before` / `registry_progress_after`)
but is not used as the checkpoint-continuity oracle. This distinction matters
on fast migrations because accounting refresh can lag the already-restored
application state.
