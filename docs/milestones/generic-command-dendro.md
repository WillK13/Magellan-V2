# Milestone: generic command and Dendro runtime

## Scope

This milestone generalizes Magellan V2 beyond Python modules while preserving
its decentralized ownership, auction, migration, reconciliation, accounting,
telemetry, and adaptive-policy layers.

Implemented:

- generic command runtime;
- Dendro application-checkpoint runtime adapter;
- process-group lifecycle and minimum process-count readiness;
- manifest-based multi-rank checkpoint validation;
- durable runtime launch metadata;
- configured and observed node capabilities;
- hard compatibility pruning and destination admission checks;
- Dendro-compatible multi-process validation workload;
- Boston-to-Virginia GCP validation script.

Deferred:

- SLURM;
- JAX/Orbax;
- container runtime;
- GPU checkpoint conversion;
- automatic conversion between incompatible checkpoint formats.

## Local validation

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
ruff check magellan tests
python -m compileall -q magellan
pytest -q
```

Expected:

```text
73 passed
```

## GCP daemon environment

Use the same isolated state-root name on Boston and Virginia:

```bash
export MAGELLAN_CONFIG=config/cluster.dev.json
export MAGELLAN_POLICY=config/policy.dev.json
export MAGELLAN_DATASETS=datasets
export MAGELLAN_STATE_ROOT="$PWD/runtime-state-generic-command-dendro"
export MAGELLAN_REMOTE_STATE_ROOT="$PWD/runtime-state-generic-command-dendro"
export MAGELLAN_REPOSITORY_ROOT="$PWD"
export MAGELLAN_SSH_USER=WILL
export MAGELLAN_TASK_FILES=""
export PYTHONUNBUFFERED=1

unset MAGELLAN_TEST_FORCE_ACTIVATION_RESPONSE_LOSS
unset MAGELLAN_AUCTION_STRATEGY
```

Set `MAGELLAN_NODE_ID=boston` on Boston and
`MAGELLAN_NODE_ID=virginia` on Virginia, then start:

```bash
python -m uvicorn magellan.api.app:app --host 0.0.0.0 --port 8040
```

## Two-node validation

From a second Boston terminal:

```bash
export BOSTON_API=http://127.0.0.1:8040
export VIRGINIA_API=http://10.162.0.2:8040
export VIRGINIA_SSH=WILL@10.162.0.2

scripts/validate_two_node_generic_command_dendro.sh
```

The validator proves:

1. both nodes discover x86_64/Linux runtime capabilities;
2. configured capabilities match observed capabilities;
3. an ARM-only task bid is rejected as incompatible without fairness credit;
4. an arbitrary command task launches with durable command metadata;
5. the Dendro-compatible workload creates one launcher and two rank workers;
6. live telemetry observes at least three processes;
7. the multi-file checkpoint manifest validates every rank file;
8. migration stops the complete Boston process group;
9. Virginia appends the Dendro resume arguments;
10. Virginia records `resumed_from_checkpoint=true`;
11. progress advances beyond the pre-migration step;
12. the destination checkpoint and runtime metadata are durable.

Expected final line:

```text
ALL TWO-NODE GENERIC COMMAND AND DENDRO CHECKS PASSED
```

## Connecting the real Dendro-GR executable

Replace the validation harness command in a task definition with the deployed
Dendro launcher, for example:

```json
{
  "adapter": "dendro",
  "command": ["mpirun", "-np", "2", "/absolute/path/to/BSSN_GR"],
  "arguments": ["/absolute/path/to/pars/q1.par.json"],
  "resume_arguments": ["<actual Dendro restart arguments>"],
  "minimum_process_count": 3
}
```

The exact BSSN_GR restart flags and checkpoint filenames must match the Dendro
build deployed on both nodes. Add `mpirun`, an OpenMPI version range, and the
`mpi` feature to the task compatibility requirements and node capability
records once OpenMPI is installed.
