# Magellan V2 Telemetry Contract

## Purpose

The telemetry subsystem supplies measured inputs to the existing Magellan
scoring, accounting, migration, and task-auction paths. It does not create a
second scheduler. Configured values remain mandatory fallbacks, and every
consumer can determine whether a value was measured, stale, or unavailable.

## Durable state

Each peer stores local observations atomically in:

```text
<state-root>/control/telemetry.json
```

The document contains task records, directed edge records, and migration
calibration records. Records survive daemon restarts but are never replicated
as task ownership state: task telemetry is meaningful only at the node where it
was observed, while each source node owns measurements for its outgoing edges.

## Task telemetry

Linux subprocess workloads are started in independent process groups. The
procfs provider aggregates all members of that group and reports:

- CPU time and derived utilization;
- resident memory;
- process count and leader state;
- current checkpoint-directory size;
- progress rate and remaining-time estimate from the durable task registry.

Power providers are selected in this order:

1. Intel RAPL package power, allocated by task CPU share when available;
2. configured full-load task power scaled by procfs CPU utilization;
3. configured task power fallback when a first sample or platform limitation
   prevents utilization measurement.

PUE remains a site property and is applied by runtime accounting after task
power is selected.

## Directed edge telemetry

Each node periodically measures HTTP round-trip time to every peer. Real
checkpoint transfers provide effective throughput observations. Both signals
are smoothed using a configurable exponential moving average and are attached
to directed edges, because Boston-to-Virginia and Virginia-to-Boston can differ.

A fresh transfer observation replaces configured bandwidth in the next
migration estimate. A fresh latency observation replaces configured latency.
When either observation exceeds its freshness threshold, the graph immediately
returns to the configured value.

## Migration calibration

Successful migrations record phase durations:

- source stop plus checkpoint validation;
- checkpoint transfer;
- destination runtime restore;
- activation request duration;
- total stop-the-world downtime.

Fresh checkpoint and restore EMAs replace the static pause/resume overheads in
future migration estimates for that directed edge. The original policy values
remain the stale-data fallback.

## Freshness and confidence

Telemetry views expose:

```text
fresh | stale | unavailable
```

Task power also exposes a source and confidence. The scheduler never silently
uses stale data. Candidate details and task bids carry the effective power and
network sources so experimental results can be audited.

## Relationship to adaptive weights

This milestone changes model inputs, not the objective weights. The future
`adaptive-policy-normalization` branch will consume these measured inputs for
budget slack, deadline slack, carbon opportunity, and rolling normalization.
Hard cost, deadline, and resource constraints will continue to override any
adapted weight vector.
