# Stage 3A.2 — Live topology-derived edge calibration

## Goal

Make WAN calibration a normal autonomous Magellan capability rather than an
experiment-only seven-node preload.

## Behavior

- Each daemon derives peers from the loaded cluster membership; no scheduler path
  assumes seven nodes or 42 edges.
- Startup/periodic telemetry probes every outgoing peer. RTT is sampled frequently;
  bounded bandwidth probes are refreshed on their own interval.
- Before autonomous or operator-triggered migration scoring, unseen/stale candidate
  edges are refreshed live. Probe failure leaves the existing conservative configured
  fallback in place.
- Bandwidth probes and observed migrations update the same timestamped edge EMA.
- Measured bandwidth has end-to-end semantics. Migration prediction uses bytes / measured
  throughput without adding RTT again; configured fallback bandwidth retains the
  bytes / bandwidth + latency model.
- `POST /telemetry/refresh` forces all outgoing edges for the node; the cluster preflight
  script invokes that topology-derived mechanism for reproducible experiment starts.
- Per-source bandwidth refresh is serialized to avoid probe self-contention.

## Scaling

The current seven-node deployment has 7*6 = 42 directed edges. If an eighth node is
added to cluster membership and runs the daemon, the same code sees 8*7 = 56 directed
edges and calibrates the new links without scheduler/model changes.
