# Stage 4A.2 — Workload + Migration Calibration

Stage 4A.2 characterizes the workloads used in the paper on the final seven-node
`e2-highmem-2` cluster and validates predicted migration overhead against actual
end-to-end Magellan migrations.  It is experiment instrumentation only; it does
not alter scheduling or migration semantics.

## Inputs

- A checksum-valid Stage 4A.1 bundle containing all 42 directed WAN edges.
- Seven idle measurement-mode workers on the same experiment commit.
- Dendro/OpenMPI pre-staged on all seven workers.
- The same local CPU LLM snapshot and Python dependency versions pre-staged on
  all seven workers with `scripts/provision_llm_all_gcp_nodes.py`.

## Representative WAN regimes

The campaign deterministically selects three directed edges from Stage 4A.1:

- **short**: highest measured median bandwidth,
- **medium**: edge nearest the directed-mesh median bandwidth,
- **long**: lowest measured median bandwidth.

The selected edges are recorded in `representative_edges.json`; they are never
hard-coded after seeing workload results.

## Workload matrix

Checkpointable benchmark workloads are run as:

- N-body: small / medium / large,
- JSON: small / medium / large,
- Matmul: small / medium / large.

Small, medium, and large map to short, medium, and long WAN regimes respectively.
Each case profiles CPU, RSS memory, checkpoint bytes, progress rate, and measured
power before forcing one production-path Magellan migration.

Dendro-GR is measured at three parameter pairs:

- `BSSN_MAXDEPTH=8`, `BSSN_RK_TIME_END=0.5` on the short regime,
- `BSSN_MAXDEPTH=9`, `BSSN_RK_TIME_END=1.0` on the medium regime,
- `BSSN_MAXDEPTH=10`, `BSSN_RK_TIME_END=2.0` on the long regime.

The real LLM workload uses one identical pre-staged `distilgpt2` snapshot and is
migrated once on each WAN regime.  Model weights, tokenizer, AdamW optimizer
state, RNG state, and exact resume continuity remain validated by the existing
real-LLM migration harness.

## Measurements

Every migration records the candidate prediction and the actual migration event:

- checkpoint bytes,
- checkpoint seconds,
- transfer seconds,
- restore seconds,
- migration overhead seconds,
- total downtime seconds,
- checkpoint/transfer/restore/downtime prediction error.

Accuracy is reported descriptively.  No threshold is used to discard an
inconvenient calibration sample.

## Expected output

A complete Stage 4A.2 bundle contains 15 cases:

- 9 benchmark cases,
- 3 Dendro cases,
- 3 real LLM cases.

The parent bundle includes `workload_profiles.csv`, `migration_samples.csv`,
`case_summaries.json`, `representative_edges.json`, metadata, summary, and
checksums, plus the complete raw child bundles for every case.
