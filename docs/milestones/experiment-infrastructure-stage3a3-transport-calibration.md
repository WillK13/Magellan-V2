# Experiment Infrastructure Stage 3A.3 — transport-faithful calibration

Stage 3A.3 closes the remaining mismatch between autonomous edge probing and the
transport used by real checkpoint migration.

## Design

- Cluster membership remains topology-derived. For `N` configured nodes, each daemon
  can calibrate its `N-1` outgoing edges without node-specific code.
- HTTP `/health` probes remain the RTT/liveness signal.
- Migration throughput is measured with incompressible rsync-over-SSH traffic using
  the same timed transfer component as checkpoint migration. SSH destination setup
  is excluded from throughput and remains part of the separately calibrated migration
  overhead.
- The first bandwidth probe is small. If it completes too quickly to be informative,
  one larger follow-up is chosen adaptively and capped by policy.
- Periodic bandwidth maintenance is deliberately infrequent; unseen edges are probed
  immediately, and stale candidate edges are refreshed lazily before scoring. This
  avoids turning telemetry into a background network benchmark.
- Actual migrations and rsync probes update the same migration-transport EMA.
- Measured migration-transport bandwidth is end-to-end throughput, so RTT is not
  added again by the migration transfer-time model.

## Acceptance gate

1. Reset isolated measurement mode.
2. Force a topology-wide live refresh.
3. Run Boston→Virginia with a 10 MiB controlled checkpoint for two samples.
4. Require the first candidate to use `measured_migration_transport_ema`.
5. Re-run the directed network characterization and compare prediction error.
6. If warm transfer/downtime errors are reasonable, freeze Stage 3A and proceed to
   the compact path/size matrix, then real LLM and Dendro validation.
