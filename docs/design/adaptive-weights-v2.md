# Adaptive objective weights in Magellan V2

This document preserves the adaptive-weight design goal from Magellan V1 while defining how it should be implemented in the decentralized V2 control plane.

## V1 compatibility target

V1 described a two-stage policy:

1. Calibrate a baseline weight vector `theta* = (alpha0, beta0, gamma0)` offline over historical traces.
2. At runtime, scale that baseline using bounded signals for budget slack, time slack, and carbon opportunity, then project the result back onto the probability simplex.

V1 also proposed rolling min/max normalization so time, carbon, and cost remain numerically comparable as their ranges change.

V2 should retain those ideas, but adaptation must consume measured distributed state rather than simulator-only totals.

## Why adaptation is not enabled in this milestone

Adaptive weights are only meaningful if their inputs are reliable. Before `pause-runtime-accounting`, V2 did not have:

- continuously accumulated per-task cost;
- continuously accumulated per-task carbon;
- progress-derived remaining runtime;
- pause duration accounting;
- migration accounting that followed task ownership.

This milestone supplies those prerequisites. The fixed configured weights remain authoritative until the dedicated adaptive-policy milestone.

## Proposed V2 architecture

### Baseline weights

Store the calibrated baseline in policy configuration:

```json
{
  "weights": {
    "time": 0.47,
    "carbon": 0.27,
    "cost": 0.26
  }
}
```

Custom operator weights should disable online adaptation unless the operator explicitly opts in.

### Runtime signals

For task `j` at epoch `t`:

```text
budget_slack = (cost_cap - accumulated_cost) / cost_cap

time_slack = (deadline - now - estimated_remaining_time) / deadline_window

carbon_opportunity =
    (current_region_forecast - best_feasible_peer_forecast)
    / max(current_region_forecast, epsilon)
```

The current branch now supplies `accumulated_cost` and `estimated_remaining_time`. A later deadline milestone must supply task deadlines. A live-carbon milestone must supply forecast confidence and stale-data status.

### Bounded multipliers

A future `WeightAdapter` should compute multipliers such as:

```text
m_time   in [0.75, 1.25]
m_carbon in [0.75, 1.25]
m_cost   in [0.75, 1.25]
```

Then:

```text
raw = (
    alpha0 * m_time,
    beta0  * m_carbon,
    gamma0 * m_cost,
)

effective = project_to_simplex(raw)
```

Hard constraints always take precedence. For example, reaching a cost cap prunes migration regardless of the adapted weight vector.

### Rolling normalization

Each peer should persist a bounded rolling window or exponentially decayed extrema for the raw action estimates:

```text
time_seconds
carbon_grams
cost_usd
```

The normalization state must be versioned and attached to decision logs. Peers bidding on the same task must receive the already-scored candidate from the owner rather than independently renormalizing it with unrelated local history.

### Explainability record

Every decision should eventually record:

```json
{
  "baseline_weights": [0.47, 0.27, 0.26],
  "effective_weights": [0.41, 0.24, 0.35],
  "signals": {
    "budget_slack": 0.12,
    "time_slack": 0.44,
    "carbon_opportunity": 0.18
  },
  "normalization_version": 17,
  "pruned_actions": ["migrate:cost_cap"]
}
```

## Safety rules

- Never adapt from stale carbon or progress telemetry.
- Never allow adaptation to override hard cost, deadline, compatibility, or capacity constraints.
- Clamp all multipliers and project back onto the simplex.
- Preserve a minimum dwell period after material weight changes.
- Persist adaptation state so daemon restarts do not reset policy behavior.
- Log both baseline and effective weights for every decision.

## Planned implementation milestone

Implement adaptation after distributed reconciliation and live telemetry, in a branch such as:

```text
adaptive-policy-normalization
```

That milestone should add a pure, deterministic `WeightAdapter`, persisted rolling normalization, decision records, replay tests against V1 traces, and bounded-response tests for budget/time/carbon signals.
