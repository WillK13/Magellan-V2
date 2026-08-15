# Stage 3B.1: real LLM migration validation

This stage validates Magellan's application-checkpoint migration path with an actual
Hugging Face causal language model rather than a synthetic checkpoint payload.

The workload checkpoint contains the local Hugging Face model/config/tokenizer,
AdamW optimizer state, PyTorch RNG state, and completed training step. Every
checkpoint receives a unique checkpoint ID. On restore, the destination verifies
that model metadata and optimizer state refer to the same checkpoint ID, loads the
optimizer/RNG state, and records the source checkpoint ID in its readiness marker.

`scripts/measure_llm_migration.py` then verifies four properties for every hop:

1. The stop-induced source checkpoint ID is exactly the checkpoint ID restored by
   the destination.
2. The destination loaded optimizer state.
3. The destination resumed at exactly the source checkpoint step.
4. Training advanced after restore.

The operator migration endpoint is used only to force a repeatable validation
route. The task still traverses Magellan's normal migration candidate, bidding,
reservation, checkpoint validation, rsync transfer, activation, restore,
ownership, telemetry, and accounting mechanisms.

The default model is `distilgpt2`. LLM dependencies remain optional so normal
Magellan nodes do not need PyTorch/Transformers unless they execute an LLM task.
The measurement script has `--preflight-only` to check the selected nodes before
launching the workload.

The experiment bundle is stored under `experiments/measurements/llm-*` and
contains `metadata.json`, `summary.json`, `llm_migrations.csv`, raw per-hop JSON,
and checksums.
