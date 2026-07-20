# V1 parity closeout

This milestone completes the remaining Magellan V1-equivalent functionality that is required before the evaluation campaign. It deliberately does not add experiment orchestration, plots, seasonal results, seven-node deployment, SLURM, CRIU, GPU migration, or JAX/Orbax.

## Included

### Correct migration carbon placement

Migration emissions are now separated into four terms:

1. source checkpoint carbon, using source PUE and source forecast;
2. destination restore carbon, using destination PUE and destination forecast;
3. destination remaining-compute carbon;
4. network-transfer carbon.

Restore is no longer charged to the source grid.

### Pause-duration search

`pause.candidate_idle_seconds` defines the durations evaluated at every decision epoch. Every duration is represented as a separate `pause` candidate with its own time, forecast carbon, normalized values, and score. The selected duration is passed directly to `PauseService` through `details.idle_seconds`.

Development defaults:

```json
"candidate_idle_seconds": [0, 300, 900, 1800, 3600]
```

### Simple carbon forecast

The decision model uses only carbon samples at or before the current decision time. The default provider:

- fits a least-squares linear trend over recent samples;
- limits the trend to a configured maximum change per hour;
- clamps projected values using recent observed bounds;
- averages projections over the requested future window;
- reports freshness, confidence, slope, residual error, history size, and whether clamping occurred;
- falls back to current-value persistence when history is insufficient;
- optionally falls back to a configured regional value when samples are stale or unavailable.

Runtime accounting still uses realized trace values for elapsed execution. Forecasts are used for decisions about future execution.

API:

```text
GET /carbon/forecast/{node_id}
```

Query parameters:

```text
horizon_seconds
start_offset_seconds
```

### Complete calibration utility

`magellan.policy.calibration` now provides:

- exact simplex-grid generation;
- configurable grid step;
- hard cost-cap pruning;
- optional deadline pruning;
- time/carbon/cost normalization over feasible candidates;
- deterministic ranking and tie breaking;
- exact grid-coverage validation;
- selected baseline writing into a policy JSON file.

Generate the paper's 0.02 simplex:

```bash
python -m magellan.policy.calibration \
  --step-size 0.02 \
  --generate-grid-output /tmp/magellan-grid.json
```

Select a baseline and write a usable policy:

```bash
python -m magellan.policy.calibration \
  --input config/policy-calibration.example.json \
  --cost-cap-usd 10 \
  --deadline-seconds 100000 \
  --output /tmp/calibration-result.json \
  --policy-template config/policy.dev.json \
  --policy-output /tmp/policy.calibrated.json
```

The experiment runner that produces the candidate totals is intentionally deferred. The calibration engine is complete and ready to consume those totals.

### Real Dendro-GR integration contract

The `dendro` runtime can now handle checkpoint layouts produced by a real MPI solver rather than requiring the mock workload's prebuilt manifest.

`dendro_options.checkpoint_discovery` supports:

- recursive file globs;
- simulation-step extraction using a regex;
- rank extraction using a regex;
- expected file and rank counts;
- a stability interval to avoid partially written checkpoints;
- selection of the newest complete step;
- atomic manifest generation;
- per-file SHA-256 hashes.

`dendro_options.progress` supports:

- parsing the latest numerical step from the solver log;
- writing Magellan's standard `progress.json`;
- optional total-step configuration for remaining-time estimation.

`dendro_options.completion` can synthesize Magellan's standard completion marker when the solver exits cleanly. An optional success regex prevents a zero exit code from being mistaken for a successful numerical run when the solver's log does not contain its expected completion message.

`resume_arguments` can contain:

```text
{checkpoint_directory}
{checkpoint_manifest_file}
{checkpoint_step}
```

Until the first complete checkpoint exists, normal scheduler epochs still evaluate `continue` and all pause-duration candidates, but migration is treated as hard-infeasible. The scheduler does not skip local carbon-aware decisions merely because a newly started application has not checkpointed yet.

The destination starts only from a complete manifest. The adapter also exports:

```text
MAGELLAN_DENDRO_RESUME
MAGELLAN_DENDRO_CHECKPOINT_DIRECTORY
MAGELLAN_DENDRO_CHECKPOINT_STEP
```

Start from:

```text
config/submissions/dendro-bssn-template.json
```

Replace all `<SET_...>` values with the actual BSSN_GR executable, parameter file, restart flags, checkpoint naming patterns, and progress-log expression. Then validate it:

```bash
python scripts/validate_real_dendro_definition.py \
  --definition config/submissions/dendro-bssn-real.json
```

To test checkpoint discovery against an existing output directory:

```bash
python scripts/validate_real_dendro_definition.py \
  --definition config/submissions/dendro-bssn-real.json \
  --checkpoint-directory /path/to/checkpoint
```

The actual Dendro-GR binary and parameter set are external to this repository, so an end-to-end numerical-resume test must be run after those artifacts are installed.

## Two-node validation

Start Boston and Virginia with the same clean state-root name, then run from Boston:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2
export STATE_ROOT_NAME=runtime-state-v1-parity-closeout

scripts/validate_two_node_v1_parity_closeout.sh
```

Expected final line:

```text
ALL TWO-NODE V1 PARITY CLOSEOUT CHECKS PASSED
```

The script checks forecasting, pause search, corrected migration carbon, grid generation, policy writing, Dendro checkpoint discovery/progress parsing, and the existing Boston-to-Virginia generic-command/Dendro lifecycle.

## Deferred

- experiment and results generation;
- seasonal and 100-task evaluation;
- SLURM adapter;
- CRIU;
- JAX/Orbax heterogeneous ML checkpointing;
- GPU migration;
- live ElectricityMaps ingestion;
- seven-node production deployment.
