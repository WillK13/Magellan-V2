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
