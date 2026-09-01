# Stage 4D.4 — Measured-capacity arbiter-policy evaluation

## Question

When several migration bids are **all individually resource-feasible**, but the
destination can accept only a subset, how do Magellan's destination-arbiter
policies differ?

Stage 4D.2 and Stage 4D.3 primarily expose hard resource unavailability. A task
often loses because the destination is simply full. That is useful for capacity
evaluation, but it does not isolate the arbiter.

Stage 4D.4 creates controlled feasible contention and invokes the production
`rank_bids` implementation directly.

## Resource grounding

The experiment inherits the canonical Stage 4D.3 → 4D.2 → 4D.1 provenance.

The destination is a measured 2-vCPU / 16002-MB node. A
`benchmark-json-medium` task requests the frozen Stage 4D.1 p95 vector
(approximately 0.997 CPU core and 13 MB).

One measured benchmark remains resident. The residual physical capacity can
therefore accept exactly one additional measured benchmark:

- first contender fits;
- after reserving that contender, a second contender does not fit.

No synthetic task-slot limit is used.

Checkpoint size and power are loaded from the frozen Stage 4A.2/4A.3/4A.4
calibration.

## Policies

The canonical comparison is:

1. `lowest_score`
2. `shortest_remaining`
3. `longest_remaining`
4. `credit_fair`
5. `highest_regret`

These are looked up from the installed `AuctionStrategy` enum by value, so the
experiment fails clearly if the production policy set changes.

## Controlled ranking attributes

To isolate the arbiter, all five fixed-cohort bidders use the same measured
benchmark resource vector. Candidate score, remaining work, and opportunity
loss are deliberately orthogonal controlled inputs:

| task | score | remaining fraction | opportunity loss |
|---|---:|---:|---:|
| A | 0.10 | 0.60 | 0.05 |
| B | 0.20 | 0.20 | 0.10 |
| C | 0.30 | 0.50 | 0.90 |
| D | 0.40 | 0.35 | 0.30 |
| E | 0.50 | 0.90 | 0.70 |

These are **not** presented as production carbon-score measurements. They are
controlled test inputs chosen so score, shortest remaining, longest remaining,
and highest regret have distinguishable preferences.

## Experiment 1: fixed cohort

Five bidders compete for one measured residual admission.

After the winner's reservation is released, the remaining bidders compete
again. This repeats until all five are admitted.

Outputs include:

- first winner;
- full admission order;
- mean wait rounds;
- maximum wait rounds;
- per-round credit evolution;
- total rejections.

## Experiment 2: starvation stream

One persistent bidder has a worse candidate score than a fresh challenger
arriving every round.

Each round again exposes exactly one measured residual benchmark admission.

This controlled liveness test records whether and when each policy admits the
persistent task. Under `credit_fair`, every rejection applies the production
configured credit increment. PASS is not tied to a preferred outcome: if the
configured credit policy does not rescue the task, that is itself a result to
analyze.

## Scope

This is an arbiter microbenchmark, not another end-to-end carbon replay.

Stage 4D.3 already provides the measured utilization/carbon curve. Stage 4D.4
isolates the destination-side resource-allocation mechanism so differences among
ranking policies are not obscured by annual traces, carbon forecasting, or
hard-infeasible bids.

## Outputs

- `auction_events.csv`
- `fixed_cohort_summary.csv`
- `starvation_summary.csv`
- `metadata.json`
- `summary.json`
- `checksums.sha256`
