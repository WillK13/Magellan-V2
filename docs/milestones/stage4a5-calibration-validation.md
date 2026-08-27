# Stage 4A.5 — Calibration/model validation

Stage 4A.5 is the final calibration gate before Stage 4B. It does not tune the
models against the validation measurements. It checks whether the already
frozen Stage 4A.1–4A.4 evidence generalizes well enough for the comparison
experiments.

## Held-out runtime-transfer test

Stage 4A.4 derives one regional slowdown factor per node from
`benchmark-matmul-medium`. Stage 4A.5 predicts a held-out workload as:

`Boston class median runtime × Stage 4A.4 node slowdown factor`.

The default physical validation matrix is deliberately compact but covers a
second synthetic CPU workload, the real MPI/Dendro workload, and DistilGPT2:

- nodes: Boston, South Australia, Ethiopia, Virginia;
- workloads: `benchmark-json-medium`, `dendro-r9-t1p0`, `llm-distilgpt2`;
- repetitions: 2 per workload/node pair;
- total: 24 physical runs.

The single-factor model is accepted only if the overall result, every held-out
class, and every validation node satisfy both predeclared absolute-percent-error
gates: median <= 20% and p95 <= 35%. If the gate fails, the measurements remain
valid; Stage 4B must not use the universal slowdown factor and should collect
workload-family-specific or direct per-node factors instead.

## Existing calibration evidence

The bundle also snapshots descriptive evidence from the frozen Stage 4A.1–4A.4
bundles. Stage 4A.2 migration accuracy is stratified by checkpoint scale so the
large percentage error of ~250-byte synthetic checkpoints is not conflated
with Dendro or LLM migration accuracy.

DistilGPT2 is included so the Stage 4B runtime model is not generalized to ML
without evidence. Because the model snapshot is an untracked experiment asset,
provision the identical `experiment-assets/models/distilgpt2` directory to each
validation node before the run. The static child harness verifies the model path
and at least 2.5 GiB of free disk before starting each LLM task.
