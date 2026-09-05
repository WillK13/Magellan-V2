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

After the eight benchmark/LLM jobs reach progress, the runner launches the three
unchanged Dendro jobs and immediately searches for one **direct all-live witness**.
The witness is collected with concurrent `ps`/procfs session snapshots over SSH
on all seven VMs, rather than relying on Magellan's 5-second telemetry-store
refresh. It records:

- per-task real process-session CPU percentage;
- per-task real RSS memory;
- process count and leader state;
- registry status/progress and PID;
- a same-run per-node Magellan reservation/capacity snapshot.

## PASS criteria

- 11/11 real workloads launch and appear simultaneously in one direct procfs witness;
- class mix is exactly 4 benchmark / 4 LLM / 3 Dendro;
- all seven node layouts are frozen Stage 4D.1 maximal packings;
- every task has a non-zombie session, positive real RSS, and real CPU evidence in that witness;
- every node's live Magellan reservation ledger matches the frozen request sum;
- no node exceeds its Stage 4D.1 effective capacity;
- all eleven witness rows are captured in the same physical-packing round;
- all eleven jobs stop or complete cleanly after measurement.

Stage 5E.3 subsequently introduces real-workload destination contention. Stage
5E.4 enables autonomous scheduling and compares policies.

## Short Dendro lifetime and direct live witness

The canonical `dendro-r9-t1p0` calibration case is much shorter-lived than the
DistilGPT-2 startup path. Stage 5E.2 therefore warms the eight benchmark/LLM
processes first, then launches the three unchanged Dendro r9/t1 processes and
immediately attempts a concurrent direct-process witness across all seven VMs.
This preserves the exact Stage 4D.1 workload class and reservation vector instead
of extending Dendro's simulation horizon just to make the experiment easier.

A witness only succeeds when every task's real session has at least one process,
a non-zombie leader, and positive RSS. CPU and RSS are read directly from the
remote process table, so stale Magellan telemetry cannot make a completed MPI
launcher look physically alive. The experiment does **not** require the shortest
calibration workload to remain active for an arbitrary 10-second window; Stage
5E.1 already validates workload progress and checkpoint/resume correctness.

The existing daemon completion-reconciliation latency remains a separate
correctness-hardening item before Stage 5E.3.
