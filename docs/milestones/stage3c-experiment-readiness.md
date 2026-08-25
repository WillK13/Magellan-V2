# Stage 3C: experiment-readiness extensions

Stage 3C is the final bounded implementation milestone before Stage 4 data
collection. It adds only the two changes requested for heterogeneous workload
experiments.

## 3C.1 Seeded heterogeneous workload population

`scripts/populate_workload.py` generates a reproducible population from a seed
and weighted mix. Built-in checkpointable CPU workloads are `nbody`, `json`,
and `matmul`. Each outer benchmark iteration is a durable progress unit, so the
workloads can be paused, migrated, restored, and compared under identical
population seeds.

The generator can also include real Dendro and an externally supplied LLM task
definition. Dendro variants select `BSSN_MAXDEPTH` and `BSSN_RK_TIME_END` per
task. `scripts/run_real_dendro_bssn.py` exposes these as `--resolution` and
`--time-end`, plus a guarded `--set KEY=VALUE` mechanism for additional scalar
parameters.

Example small-workload population:

```bash
python scripts/populate_workload.py \
  --count 50 \
  --seed 42 \
  --mix nbody=0.35,json=0.30,matmul=0.35 \
  --mean-interarrival-seconds 30 \
  --population-id stage4-mixed-50
```

Example including Dendro:

```bash
python scripts/populate_workload.py \
  --count 20 \
  --seed 42 \
  --mix nbody=0.3,json=0.2,matmul=0.2,dendro=0.3 \
  --dendro-solver /home/WILL/dgr-build/BSSN_GR/bssnSolver \
  --dendro-parameter-template /home/WILL/q1-magellan-magellan.toml \
  --dendro-resolutions 8,9,10 \
  --dendro-time-ends 0.5,1.0,2.0
```

Add `--submit --start` to register and launch the generated population on the
selected initial owners. The generated manifest records the population seed,
workload type, variant, owner, and scheduled arrival offset.

## 3C.2 Resource-vector destination admission

The destination arbiter already carried task CPU, memory, GPU, and accelerator
requests and maintained a resource ledger. Stage 3C completes the transition by
making the scalar task-count cap optional. `config/cluster.gcp.json` uses
`capacity: null`, so the seven-node experiment deployment admits tasks based on
remaining CPU/memory/GPU resources rather than an arbitrary one-task-per-node
limit.

The auction status and health endpoints expose reserved resources, available
resources, and `resource_busy_fraction`, defined as the dominant reserved share
of configured CPU, memory, and GPU capacity. This is reservation-based rather
than instantaneous utilization, keeping admission deterministic.

The old integer `capacity` remains supported as an optional safety cap for dev
and targeted validation configurations.

## Validation

Local:

```bash
python -m pytest -q \
  tests/test_stage3c_experiment_readiness.py \
  tests/test_resource_aware_auction.py \
  tests/test_dendro_real_checkpoint_support.py
python -m pytest -q
```

GCP after deployment:

```bash
scripts/validate_two_node_stage3c.sh
```

The GCP check proves that two 1-CPU reservations can coexist on a 2-CPU node,
a third is rejected by CPU availability rather than task count, busy fraction
reaches 1.0, cancellation releases resources, and a generated benchmark task
creates real checkpoint/progress state.
