# Stage 4B — Core calibrated policy comparison

Stage 4B is the first headline policy comparison after the Stage 4A calibration
freeze. It is a checksummed **offline calibrated replay**, not another hardware
calibration campaign. No daemon or GCP service changes are required.

## Common experimental evidence

Every policy receives the same deterministic arrivals, 2024 lifecycle-carbon
traces, configured regional prices, and frozen Stage 4A evidence:

- Stage 4A.1: all 42 directed affine WAN transfer models;
- Stage 4A.2: workload-specific checkpoint payload, checkpoint time, restore
  time, and residual migration overhead;
- Stage 4A.3: workload power;
- Stage 4A.4: Boston completion runtime and per-node slowdown factors;
- Stage 4A.5: approval of the `single_node_slowdown_factor` runtime model.

The three core workloads are `benchmark-json-medium`, `dendro-r9-t1p0`, and
`llm-distilgpt2`. Their calibrated Boston completion times are multiplied by a
single default factor of 60 to represent long-running jobs while preserving the
measured working set/checkpoint size, power, and regional performance factors.
The factor is an experiment parameter, not a claim that the physical Stage 4A
runs lasted 60 times longer.

The default annual grid contains two deterministic arrivals per month (day 5 at
00:00 UTC and day 20 at 12:00 UTC), giving 24 arrival times x 3 workloads = 72
scenarios.

## Policies

The core comparison contains exactly five policies.

1. `boston_static`: immediate execution in Boston with no scheduling action.
2. `best_static`: free initial placement among all seven nodes at arrival, then
   uninterruptible execution. Candidate selection uses the configured Magellan
   time/carbon/cost weights and the realized calibrated interval metrics.
3. `gaia_carbon_time`: equation-level reproduction of GAIA's ASPLOS'24
   Carbon-Time (`--scheduling-policy carbon --carbon-policy cst_average`). It is
   fixed to Boston, may defer job start, and then runs the job uninterrupted.
4. `magellan_causal`: calls the production `evaluate_task` path at every 900 s
   epoch with the configured causal linear-trend carbon forecast and adaptive
   policy. It may continue, pause, or migrate.
5. `clairvoyant_spatiotemporal_static_oracle`: a deliberately named offline
   reference that may choose any initial node and an hourly start time up to 24
   h into the future, but may not migrate. It is not claimed to be a universal
   optimal scheduler.

For Magellan, remaining work is maintained in Boston-equivalent seconds. A node
with slowdown factor `s` needs `work * s` wall seconds, and a wall interval
`dt` completes `dt / s` units of Boston-equivalent work. This validated slowdown
changes realized progress and outcomes; it is deliberately **not injected as a
new destination-speed feature into production migration scoring**.

Candidate and realized migration timing use the Stage 4A.1 affine model
`fixed + bytes / steady_rate` plus the Stage 4A.2 workload checkpoint, restore,
and residual overhead. Static assets are treated as pre-staged on all nodes.

## GAIA reproduction boundary

GAIA Carbon-Time selects `t_start` within a queue-specific wait window to
maximize

`(C(t) - C(t_start)) / (t_start + J_avg - t)`.

The reproduction uses the paper defaults: jobs with Boston runtime <= 2 h are
short, short jobs can wait up to 6 h, and long jobs up to 24 h. `J_avg` is the
mean scaled Boston runtime of the Stage 4B classes in that queue. Candidate
start times are hourly because the source carbon traces are hourly. GAIA is
intentionally given perfect future carbon within its wait window. The bundle
records this as a policy reproduction; it does not claim that upstream GAIA
source code was executed directly on Magellan traces.

## Output and validation

The runner writes `scenarios.csv`, `outcomes.csv`, `policy_summary.csv`, detailed
`traces.jsonl`, `calibration_model.json`, `gaia_reproduction.json`, descriptive
metrics, metadata, summary, and checksums. Outcomes include raw makespan,
carbon, cost, pause/migration counts, plus ratios against `boston_static`.

The independent validator checks source-bundle checksums, the frozen 4A.5 model
gate, exact scenario/policy coverage, policy-specific action constraints, the
GAIA paper parameters, and all output checksums.
