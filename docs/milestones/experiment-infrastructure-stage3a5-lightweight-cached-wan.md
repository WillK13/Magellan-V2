# Stage 3A.5 — Lightweight cached WAN telemetry

This milestone simplifies Magellan's WAN calibration path after the compact
migration matrix exposed poor extrapolation from startup-dominated small rsync
probes on long/high-RTT links.

## Runtime design

Normal scheduling does not benchmark candidate edges. Each daemon instead:

1. discovers peers from current cluster membership,
2. measures RTT in the background,
3. periodically calibrates each outgoing migration transport edge,
4. caches the timestamped edge model,
5. reuses that model in normal scheduling decisions, and
6. incorporates real migration observations into the same telemetry state.

A new or restarted daemon automatically calibrates previously unseen outgoing
edges during the background telemetry loop. Explicit refresh endpoints remain
available for experiment preflight and debugging, but synchronous refresh before
every scheduling decision is disabled by default.

## Transfer calibration

Each periodic transport calibration uses two lightweight measurements:

- a small incompressible rsync transfer to estimate fixed rsync/SSH startup cost;
- a bounded-duration SSH byte stream to estimate sustained source-to-destination
  transport throughput after the SSH connection is established.

The cached transfer estimate remains:

`transfer_seconds = fixed_transport_seconds + bytes / steady_bandwidth`

This preserves a simple affine model while avoiding inference of steady-state
throughput from two small transfers that are both dominated by TCP/SSH startup.

Production defaults refresh migration-throughput telemetry every 30 minutes and
keep measurements usable for one hour. Development/smoke configurations use
shorter intervals. Real migrations continue to refine the cached steady-state
rate.

No scheduler logic assumes seven nodes or 42 directed edges; the peer set is
always derived from cluster membership.
