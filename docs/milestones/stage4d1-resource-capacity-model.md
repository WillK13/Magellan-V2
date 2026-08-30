# Stage 4D.1 — Evidence-Backed Resource Capacity Model

## Goal

Stage 4D introduces multi-task contention and destination auctions. Before any
contention replay is run, Stage 4D.1 freezes the resource model from measurements
that already exist in the Stage 4A calibration chain. The purpose is to avoid an
experiment-only abstraction such as "two slots per node" when Magellan already has
CPU, memory, GPU, and accelerator-aware admission.

Stage 4D.1 does **not** run tasks and does **not** change production node
configuration. It produces a checksummed, auditable capacity/request bundle that
Stage 4D.2 will consume.

## Node capacity source

For every node, the resource ceiling is derived from Stage 4A.1 `hardware.json`.
For each scalar resource, the frozen experiment capacity is the component-wise
minimum of:

1. the configured Magellan resource capacity, and
2. the capability observed on the real GCP VM during Stage 4A.1 preflight.

This is deliberately conservative. It prevents the contention replay from assuming
more capacity than either the scheduler configuration or the measured host reported.

No synthetic task-count cap is introduced. The production GCP configuration uses
`capacity: null`, so Stage 4D admission is resource-vector based.

## Workload demand source

The three Stage 4B/4C headline classes use Stage 4A.3 aggregate p95 resource
measurements:

- `benchmark-json-medium`
- `dendro-r9-t1p0`
- `llm-distilgpt2`

CPU utilization is converted to cores by `cpu_p95_percent / 100`. RSS memory is
rounded upward to the next MiB. GPU demand remains zero for these CPU workloads.

The p95 is used rather than the median so the auction does not plan around a
best-case instantaneous sample.

## Why slowdown is not capacity

Stage 4A.4 slowdown factors remain a separate input. A node that finishes work more
quickly does not receive extra simultaneous CPUs in Stage 4D. Instead, its real
measured speed causes resources to become available sooner in the multi-task replay.
This avoids double-counting performance heterogeneity.

## Outputs

A Stage 4D.1 bundle contains:

- `node_capacities.csv`: configured, observed, and conservative effective resources;
- `workload_resource_requests.csv`: Stage 4A.3 p95-derived task requests;
- `homogeneous_capacity.csv`: maximum number of identical tasks that fit per node;
- `maximal_packings.csv`: all maximal mixed workload packings under the resource
  vectors;
- `metadata.json`, `summary.json`, and `checksums.sha256`.

## Canonical command

Use the corrected canonical Stage 4B bundle so the full Stage 4A provenance is
resolved from one immutable parent:

```bash
python scripts/run_stage4d1_resource_model.py \
  --stage4b-bundle "$MAGELLAN_STAGE4B_CANONICAL"
```

Then validate:

```bash
D41=$(find experiments/measurements -maxdepth 1 -type d -name 'stage4d1-*' \
  -print | sort | tail -1)
python scripts/validate_stage4d1_resource_model.py "$D41"
```

The expected pass marker is:

```text
STAGE_4D1_RESOURCE_MODEL_BUNDLE_PASS
```

## Boundary to Stage 4D.2

Stage 4D.2 will consume this frozen bundle together with Stage 4A.4 slowdown factors,
Stage 4A.1/4A.2 migration calibration, lifecycle carbon traces, and the corrected
production scoring policy. Multiple tasks will compete for destination resources via
the existing task-to-destination auction direction. Stage 4D.2 must not replace the
measured request vectors with hand-selected slot counts.
