# Stage 5E.2 — Physical heterogeneous maximal packing

Stage 5E.2 physically realizes the exact eleven-task measured-capacity `umax`
packing used by Stage 4D.2/4D.3, but with the actual checkpointable benchmark,
DistilGPT-2 CPU training, and two-rank Dendro-GR workloads on the seven real GCP
Magellan daemons.

## Frozen layout

| Node | Real workloads | Stage 4D.1 maximal packing |
| --- | --- | --- |
| Boston | 2 × benchmark-json-medium | 2 benchmark |
| California | benchmark-json-medium + llm-distilgpt2 | benchmark + LLM |
| South Australia | 2 × llm-distilgpt2 | 2 LLM |
| Nepal | dendro-r9-t1p0 | 1 Dendro |
| Ethiopia | dendro-r9-t1p0 | 1 Dendro |
| France | dendro-r9-t1p0 | 1 Dendro |
| Virginia | benchmark-json-medium + llm-distilgpt2 | benchmark + LLM |

Totals: 4 benchmark, 4 LLM, 3 Dendro, 11 real processes/jobs. The frozen
Stage 4D.1 resource vectors reserve approximately 12.3702 CPU cores out of the
14 effective cluster cores (88.36%).

## Methodology

Placement is fixed and every run is labeled `scheduler_mode=operator_only`.
This stage does not test carbon-policy destination selection. It isolates the
physical validity of the Stage 4 resource model.

The runtime/checkpoint definitions are the same workload forms already validated
by Stage 5E.1. Only the Magellan `resource_request` declaration is replaced with
the exact Stage 4D.1 p95 request derived from Stage 4A.3, so the live reservation
ledger is evaluated against the same evidence-backed vectors used in replay.

After all eleven jobs are simultaneously running and have made progress, the
runner samples:

- per-task process-group CPU utilization;
- per-task RSS memory;
- process counts, progress, and checkpoint bytes;
- per-node Magellan reserved CPU/memory/GPU;
- resource busy fraction and remaining capacity.

## PASS criteria

- 11/11 real workloads launch and are simultaneously `RUNNING`;
- class mix is exactly 4 benchmark / 4 LLM / 3 Dendro;
- all seven node layouts are frozen Stage 4D.1 maximal packings;
- every task has real process CPU and RSS telemetry;
- every node's live Magellan reservation ledger matches the frozen request sum;
- no node exceeds its Stage 4D.1 effective capacity;
- no workload leaves `RUNNING` during the profile window;
- all eleven jobs stop or complete cleanly after measurement.

Stage 5E.3 subsequently introduces real-workload destination contention. Stage
5E.4 enables autonomous scheduling and compares policies.

## Short Dendro lifetime and live-process gate

The canonical `dendro-r9-t1p0` calibration case is much shorter-lived than the
DistilGPT-2 startup path. Stage 5E.2 therefore warms the eight benchmark/LLM
processes first, then launches the three unchanged Dendro r9/t1 processes and
starts the physical co-location profile as soon as all eleven process sessions
are genuinely live. The default profile window is 10 seconds with 2-second
sampling. This preserves the exact Stage 4D.1 workload class and reservation
vector instead of extending Dendro's simulation horizon just to make the test
pass.

Registry `RUNNING` state alone is not accepted as liveness evidence. A workload
must have at least one procfs process, a non-zombie leader state, and positive
RSS at steady-state and throughout every physical profile sample. This matters
because operator-only tasks are reconciled at scheduler epochs; a completed MPI
launcher can otherwise remain represented as `RUNNING` until reconciliation.
