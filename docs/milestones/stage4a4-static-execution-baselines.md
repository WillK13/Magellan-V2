# Stage 4A.4 — Static execution baselines

Stage 4A.4 measures true, scheduler-isolated natural completion before the main
Magellan comparison experiments.

## Canonical-node completion matrix

The 13 Stage 4A.3 workload classes run to natural completion on Boston for three
independent trials. Synthetic benchmark iteration counts and the DistilGPT2 step
count are derived from the validated Stage 4A.3 median progress rates to target
approximately 100 seconds of finite execution. The real Dendro BSSN-GR variants
retain their physical `(resolution, time_end)` settings unchanged.

Every run uses `scheduler_mode=operator_only`; no pause or migration is allowed.
The bundle records wall time, Magellan accumulated runtime, cost, carbon,
progress, and telemetry samples through completion.

## Seven-node equivalence check

`benchmark-matmul-medium` is repeated three times on every final-hardware node.
Boston's three canonical trials are reused, so the equivalence phase adds only
18 physical runs rather than rerunning Boston. Per-node median runtime and
slowdown relative to Boston are reported descriptively. This tests whether the
identical `e2-highmem-2` deployment needs a regional performance correction in
later static/oracle replay baselines.

Stage 4A.4 does not tune Magellan and does not choose a best region. It provides
measured completion/runtime evidence for Stage 4A.5 validation and later baseline
replay under historical carbon and regional price traces.
