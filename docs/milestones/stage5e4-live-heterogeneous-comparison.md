# Stage 5E.4 — live heterogeneous static-vs-Magellan comparison

Stage 5E.4 is the bounded live-system companion to the frozen Stage 4D.3
winter/u75 replay. It does not introduce a new scheduler or synthetic workload.

## Frozen population

The experiment reads the exact `winter/u75` layout from the canonical Stage
4D.3 bundle (`winter-20240105T0000Z-layout0`) and physically launches its nine
real workloads:

- 3 × `benchmark-json-medium`
- 3 × `llm-distilgpt2`
- 3 × `dendro-r9-t1p0`

Admission uses the exact Stage 4D.1 measured p95 resource requests. The achieved
initial request is approximately 75.8% of the 14-core cluster.

Stage 5A explicitly installs `config/policy.stage5e4.json`, which is byte-for-byte
equivalent in settings to `config/policy.prod.json` except for the trace-clock
start: the experiment policy anchors to the frozen winter arrival date,
`2024-01-05T00:00:00Z`. The production policy remains unchanged. All scoring
weights, migration thresholds, auction strategy (`lowest_score`), recovery,
telemetry, and resource capacities remain unchanged.

## Policies

Two sequential trials use the same task layout and fixed wall-clock measurement
window:

1. `static_initial_layout`: no scheduler evaluation is triggered.
2. `magellan_lowest_score`: one synchronized production scheduler epoch is
   triggered for the six benchmark/LLM tasks. The three exact Dendro workloads
   remain real physical background load during the decision instant.

The daemons are restarted before each trial so their trace clocks reset to the
same frozen Jan-5 source window. Tasks remain `operator_only` to prevent the
900-second background scheduler from racing the controlled measurement.
Explicit `/tasks/<id>/evaluate` calls still execute the exact production
scoring, bidding, arbitration, checkpoint, rsync, activation, and ownership
paths.

Credit fairness is intentionally not included here. Its mechanism was isolated
in Stage 4D.4; changing the live destination-arbiter strategy would require a
different daemon deployment and would confound the clean static-vs-production
comparison.

## Physical evidence

Before each trial begins, Stage 5E.4 requires one direct process-session witness
showing all nine actual workloads simultaneously alive. LLM startup is completed
before the short Dendro jobs are launched so model initialization cannot consume
Dendro's useful lifetime before the witness.

During the fixed trial window, the harness samples all seven daemons for:

- ResourceLedger CPU/RAM reservations and remaining capacity;
- live task CPU/RSS telemetry;
- physical owner count;
- capacity violations.

At the end of each window it freezes authoritative task-accounting deltas
from the pre-window baseline, final ownership, scheduler decisions, bids, migrations,
downtime, progress, carbon,
cost, and cleanup evidence.

## PASS criteria

The experiment passes when:

- both trials use the exact same frozen nine-task layout and 3/3/3 class mix;
- both obtain a 9/9 direct pre-trial live-process witness;
- the static trial produces zero scheduler decisions, bids, or migrations;
- the Magellan trial produces exactly six controlled benchmark/LLM scheduler
  decisions, at least one bid, and at least one successful real migration;
- no migration fails;
- no sampled node exceeds configured resource capacity;
- ownership converges at the end of each trial;
- all nine tasks are completed/stopped cleanly per trial;
- both trials accrue positive lifecycle carbon and cost over the fixed window.

Carbon savings are reported, not assumed as a correctness condition. A short
live trial remains scientifically valid even if migration overhead outweighs
compute-carbon savings in that window.
