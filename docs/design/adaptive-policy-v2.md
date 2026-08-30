# Adaptive policy and rolling normalization in Magellan V2

## Purpose

Magellan first selects a stable baseline objective policy from historical runs and then adapts that baseline during a live task. Adaptation is task-local and follows the task when ownership migrates.

The baseline remains the configured convex weight vector:

```text
θ₀ = (α₀, β₀, γ₀)
α₀ + β₀ + γ₀ = 1
```

The three objectives are time, operational carbon, and monetary cost.

## Offline calibration

`python -m magellan.policy.calibration` consumes aggregate historical outcomes for candidate weight vectors. Hard cost and deadline constraints are applied before candidates are normalized and ranked. The selected candidate can then be copied into `policy.weights`.

Example:

```bash
python -m magellan.policy.calibration \
  --input config/policy-calibration.example.json \
  --cost-cap-usd 10 \
  --deadline-seconds 100000
```

## Runtime signals

For every owned running task, each scheduler epoch derives:

- **Budget slack**: remaining budget divided by the task cost cap.
- **Deadline slack ratio**: wall/trace time available before the deadline divided by estimated remaining work.
- **Carbon opportunity**: relative carbon advantage of the best feasible migration over the best local action.
- **Telemetry confidence**: confidence attached to the effective live power estimate.

Hard constraints are evaluated before adaptation. In particular, a migration that violates the task cost cap is removed before scoring and cannot be restored by a large carbon weight.

## Bounded multipliers

The baseline is scaled by bounded multipliers:

```text
α' = α₀ · m_time
β' = β₀ · m_carbon
γ' = γ₀ · m_cost
```

The default bound is ±25%, so each multiplier remains in `[0.75, 1.25]`. The scaled vector is projected back onto the simplex:

```text
(α_t, β_t, γ_t) = normalize(α', β', γ')
```

The current implementation increases:

- `m_time` as deadline slack is exhausted;
- `m_cost` as budget slack is exhausted;
- `m_carbon` when a high-confidence lower-carbon spatial opportunity exists.

## Rolling normalization

Per-epoch min/max normalization can make identical physical values receive very different normalized scores as candidate sets change. V2 therefore stores recent ranges for each objective across scheduler epochs.

For each task and objective, the policy store retains the minimum and maximum observed in the most recent `rolling_window_epochs`. Time, carbon, and monetary cost are non-negative quantities with a meaningful physical zero, so production scoring uses a zero-anchored rolling range: the lower normalization bound is `0` and the upper bound is the rolling maximum. This preserves adaptation to changing scale without expanding a narrow absolute spread (for example, a few cents of compute price) to the entire `[0,1]` interval. New observations enter the window before the current epoch is scored. The legacy rolling min/max behavior remains available through `adaptive.normalization_zero_anchor=false` for replay/debugging.

## Durable state

The state file is:

```text
<state-root>/control/adaptive-policy.json
```

Each task stores:

- baseline and effective weights;
- bounded multipliers;
- current signals;
- rolling time/carbon/cost ranges;
- decision count;
- most recent decision;
- bounded decision history.

The state is transferred in migration activation requests and ownership anti-entropy snapshots. A destination therefore continues the same decision count and normalization window.

## Explainability

Every decision record includes:

- selected action and destination;
- selected score;
- baseline and effective weights;
- each multiplier;
- budget/deadline/carbon signals;
- normalization bounds;
- hard-constraint status;
- scheduler reason.

The API exposes this through:

```text
GET /policy
GET /policy/tasks
GET /policy/tasks/{task_id}
POST /policy/tasks/{task_id}/reset
```

## Relationship to destination auctions

Adaptive weights determine how a task values continue, pause, and each destination migration. They do not reverse the auction direction. Tasks still bid for destination capacity. A destination such as France then ranks the feasible incoming task bids using its configured auction strategy.
