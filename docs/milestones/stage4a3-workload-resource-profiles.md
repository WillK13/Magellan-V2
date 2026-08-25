# Stage 4A.3 — Workload Resource Profiles

Stage 4A.3 measures steady-state resource demand for the workloads used by the
Stage 4 experiments on the final `e2-highmem-2` hardware. It deliberately does
**not** repeat WAN or migration calibration from Stage 4A.1/4A.2.

## Methodology

- Run on one canonical final-hardware node (`boston` by default). Node-to-node
  execution variability belongs to the Stage 4A.4 static baselines.
- Every run is labeled `scheduler_mode=operator_only`.
- No operator migration is requested; the task is stopped after the profile
  window.
- Run three independent trials per workload class by default.
- Sample aggregate Linux workload-session telemetry. This includes MPI children
  whose process groups differ from the `mpirun` leader but whose session ID is
  the Magellan workload leader PID.

The workload matrix has 13 classes:

- N-body small / medium / large,
- JSON small / medium / large,
- Matmul small / medium / large,
- Dendro `(MAXDEPTH,time_end)` = `(8,3.0)`, `(9,1.0)`, `(10,2.0)`,
- one real CPU `distilgpt2` causal-LM training profile.

Each profile records process count, aggregate CPU, RSS, checkpoint-directory
footprint, progress rate, estimated remaining time, and the existing
utilization-based power estimate. `checkpoint_bytes` is a filesystem footprint;
Dendro may transiently expose one versus two rotating checkpoint generations.
Use Stage 4A.2 migration events for actual transferred checkpoint size.

## Expected bundle

With the default three trials, a complete bundle contains 39 passed profile-only
runs across 13 classes. The parent bundle contains:

- `profile_runs.csv`,
- `profile_classes.csv`,
- `case_summaries.json`,
- `metadata.json`,
- `summary.json`,
- `checksums.sha256`,
- complete child measurement bundles under `measurements/`.

Validate with:

```bash
python scripts/validate_stage4a3_profiles.py experiments/measurements/<stage4a3-id>
```
