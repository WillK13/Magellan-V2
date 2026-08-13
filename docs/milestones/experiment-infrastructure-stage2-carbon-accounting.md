# Experiment Infrastructure Stage 2.1: carbon accounting

Stage 2.1 makes the carbon-intensity series an explicit experiment input instead of
an implicit constant.

## Policy

- `direct` uses `Carbon intensity gCO₂eq/kWh (direct)`.
- `lifecycle` uses `Carbon intensity gCO₂eq/kWh (Life cycle)`.
- Existing Magellan daemon behavior remains backwards-compatible: without an
  override, daemons continue to use `direct`.
- Offline NSDI comparison tooling defaults to `lifecycle` because it avoids treating
  zero-at-operation generation as literally zero-footprint electricity.
- Direct operational intensity remains a first-class sensitivity mode.

## Reproducibility

The selected metric and exact CSV column are written into every Stage-2 comparison
manifest. Daemons expose the active metric through `/health` and
`/experiment/events/status`, and recorded experiment runs reject a seven-node cluster
whose daemons disagree on the carbon metric.

For measured NSDI runs, the experiment service mode will explicitly set
`MAGELLAN_CARBON_METRIC=lifecycle`; the default is intentionally not changed so the
v1.2 system remains backwards-compatible outside the experiment harness.
