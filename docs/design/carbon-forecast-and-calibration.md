# Carbon forecast and baseline calibration contract

## Forecast causality

A decision evaluated at time `t` may consume only samples whose timestamps are less than or equal to `t`. Forecast windows may begin later than `t`, such as destination compute after checkpoint, transfer, and restore, but their regression history remains anchored at `t`.

## Linear forecast

For recent samples `(x_i, y_i)`, where `x_i` is hours relative to the newest observation, Magellan fits an ordinary least-squares line. The slope is bounded by `maximum_change_per_hour`. Each predicted value is then clamped to a non-negative range derived from recent minimum and maximum observations plus the configured maximum hourly change.

The returned carbon intensity is the mean of samples across the requested forecast window. This is intentionally a small and explainable predictor, not an advanced grid model.

## Confidence

Confidence combines:

- the fraction of the requested history window available;
- normalized regression residual error;
- sample freshness;
- a penalty when slope or predicted values require clamping.

Insufficient history uses persistence with a lower configured confidence. Stale data returns configured fallback when one exists; otherwise it returns persistence with zero confidence.

## Decision versus accounting

Forecast carbon is used only for prospective action scoring. Runtime accounting uses realized carbon values for intervals that have already elapsed.

## Baseline calibration

The simplex generator uses exact decimal arithmetic to require that the step divides one. A step of 0.02 produces all non-negative time/carbon/cost combinations summing to one.

Hard budget and deadline constraints are applied before normalization and scoring. Normalization is computed only across feasible candidates. Ranking is deterministic, including ties.

Calibration output can update the `weights` object of an existing policy file atomically. Online adaptation then treats those calibrated values as its baseline and applies the existing bounded multipliers and simplex projection.
