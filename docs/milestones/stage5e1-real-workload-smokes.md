# Stage 5E.1 — Real heterogeneous workload migration smokes

## Purpose

Stage 5E.1 bridges the frozen Stage 4A workload measurements to the hardened
seven-node Stage 5 deployment. It runs one real production-path migration for
each of the three canonical workload classes:

- `benchmark-json-medium`
- `llm-distilgpt2`
- `dendro-r9-t1p0`

No scheduler policy result is being measured here. Destinations are deliberately
operator-directed so the experiment isolates workload-specific migration
correctness.

## Hardened deployment requirement

The experiment only runs from a passed Stage 5A format-v2 bundle whose exact Git
SHA matches the local runner and all seven daemons. The daemons must use:

- lifecycle carbon;
- `config/cluster.gcp.json`;
- `config/policy.prod.json`;
- `/home/WILL/Magellan-V2/runtime-state-gcp` for local and remote state;
- zero systemd drop-ins;
- no pre-existing active tasks.

## Cases

### Benchmark

`benchmark-json-medium` runs the real checkpointable Python JSON benchmark from
Boston and migrates to California. The existing Stage 4A.2 real workload
migration harness validates checkpointed progress, rsync transfer, activation,
ownership, and resumed progress.

### LLM

`llm-distilgpt2` runs real CPU Hugging Face causal-LM training from California
and migrates to France. The checkpoint contains model weights/config,
tokenizer, AdamW optimizer state, PyTorch RNG state, and completed step. PASS
requires checkpoint-ID equality, optimizer-state reload, exact-step resume, and
continued training.

### Dendro

`dendro-r9-t1p0` runs the real upstream Dendro-GR BSSN solver with two local MPI
ranks from Virginia and migrates to Nepal. The existing Dendro adapter discovers
native BSSN checkpoint files and restores them on the destination.

## PASS

PASS requires all three child measurements to pass, exactly one real migration
per workload, and workload-specific resume validation for all three cases.
Mixed placement, physical contention, and multi-task policy behavior are
reserved for Stage 5E.2–5E.4.
