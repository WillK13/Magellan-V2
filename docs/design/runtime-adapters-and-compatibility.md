# Runtime adapters and compatibility contract

Magellan's scheduler operates on a runtime-neutral lifecycle:

- start;
- pause;
- resume;
- checkpoint validation;
- stop;
- status/reconciliation.

A runtime adapter converts a task definition into the concrete local command
that implements that lifecycle. The scheduler, bidding market, migration
journal, accounting, telemetry, and ownership protocol remain unchanged.

## Supported adapters

### `python_module`

Preserves the original behavior:

```json
{
  "adapter": "python_module",
  "module": "magellan.workloads.counter",
  "arguments": ["--checkpoint-file", "{checkpoint_file}"]
}
```

Magellan launches the module with the active interpreter.

### `command`

Runs an arbitrary executable without a shell:

```json
{
  "adapter": "command",
  "command": ["python3", "-m", "my_package.worker"],
  "arguments": ["--state", "{checkpoint_file}"]
}
```

Command and argument elements support the standard placeholders:

- `{task_id}`;
- `{task_directory}`;
- `{repository_root}`;
- `{artifacts_directory}`;
- `{checkpoint_file}`;
- `{checkpoint_directory}`;
- `{checkpoint_manifest_file}`;
- `{readiness_file}`;
- `{progress_file}`;
- `{completion_file}`;
- `{output_directory}`.

The process is launched in a new session. The leader PID is therefore also the
process-group ID, allowing pause, resume, stop, telemetry, and migration to
cover child workers rather than only the launcher.

### `dendro`

The Dendro adapter is a command adapter with an application-checkpoint restart
contract. It launches `command + arguments` on an initial run. When a valid
checkpoint has been transferred to the destination, it appends
`resume_arguments` and exports:

```text
MAGELLAN_DENDRO_RESUME=1
MAGELLAN_DENDRO_CHECKPOINT_DIRECTORY=<path>
```

The task definition must use a manifest-based checkpoint when the checkpoint
contains multiple rank files. Magellan verifies that every manifest entry is
present and has the expected size before migration or activation.

The repository includes `magellan.workloads.dendro_mock`, a Dendro-compatible
MPI-process-tree harness used to validate the lifecycle on GCP. It is not a
replacement for the BSSN_GR executable. A real deployment substitutes the
actual `mpirun` command and Dendro restart arguments while preserving the same
checkpoint, progress, readiness, and completion paths.

## Durable runtime metadata

Each task state records:

```text
runtime_adapter
launch_command
process_group_id
resumed_from_checkpoint
```

This metadata survives daemon restarts and follows ownership through the
existing task-state migration path.

## Node capabilities

Each configured location advertises a scheduling contract:

```json
{
  "architecture": "x86_64",
  "operating_system": "linux",
  "cpu_cores": 2,
  "memory_mb": 4096,
  "gpu_count": 0,
  "commands": ["bash", "python3", "rsync"],
  "runtimes": {"python": "3.11"},
  "features": [
    "local-command",
    "python-module",
    "process-group",
    "application-checkpoint",
    "dendro-adapter"
  ]
}
```

`GET /capabilities` returns both the configured contract and locally observed
architecture, OS, commands, runtimes, and features. It also reports drift.

## Task compatibility requirements

A task can declare:

```json
{
  "architectures": ["x86_64"],
  "operating_systems": ["linux"],
  "minimum_cpu_cores": 1,
  "minimum_memory_mb": 512,
  "required_commands": ["python3"],
  "required_runtimes": {"python": ">=3.11,<3.12"},
  "required_features": [
    "local-command",
    "process-group",
    "application-checkpoint",
    "dendro-adapter"
  ],
  "checkpoint_architecture_independent": false,
  "requires_same_mpi_world_size": true
}
```

Supported version comparisons are `==`, `>=`, `<=`, `>`, and `<`, including
comma-separated ranges.

## Enforcement order

Compatibility is a hard feasibility constraint:

1. The source prunes incompatible destinations before migration actions are
   scored.
2. Operator-triggered migration rejects an incompatible destination before a
   bid is created.
3. The destination auction rechecks the task against locally observed
   capabilities before resource admission.
4. The destination runtime rechecks compatibility immediately before launch.

An incompatibility never earns fairness credit because it is not resource
competition. Bid records expose:

```text
compatibility_fit
compatibility_reasons
```

Resource capacity and runtime compatibility remain separate checks.

## Deferred adapters

- SLURM remains optional and should be added only when a Magellan location
  represents a multi-node cluster managed by SLURM.
- JAX/Orbax remains a later adapter. It is intended to address structured
  training-state restoration and heterogeneous device/topology portability.
