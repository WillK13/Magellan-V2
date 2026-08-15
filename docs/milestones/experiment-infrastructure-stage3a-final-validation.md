# Stage 3A final validation matrix

This milestone freezes the Stage-3A model implementation and runs a compact real-system
validation matrix before moving to real LLM and Dendro checkpoint behavior.

The default matrix is intentionally small but heterogeneous:

- Boston -> Virginia: relatively short/fast WAN path.
- Boston -> France: medium intercontinental path.
- California -> South Australia: long intercontinental path.
- Boston -> Nepal: extreme high-RTT/low-throughput path.

Checkpoint payloads are 10 MiB, 100 MiB, and 500 MiB, with two samples per case by
default. This is 12 path/size cases and 24 real migrations. The harness raises the
per-migration timeout to one hour because 500 MiB on the slowest path can legitimately
take many minutes.

The matrix reuses the production migration path and Stage-3A live edge calibration.
It does not introduce a new predictor and it does not hard-code measured WAN values.
Every raw run remains in `migration_samples.csv` and `raw/`. Headline matrix accuracy
is computed only over samples whose migration candidate actually used the learned
migration calibration and a live measured transfer model. Cold/fallback samples are
retained and counted separately rather than discarded.

Outputs added to the controlled migration bundle:

- `matrix_summary.json`: overall calibrated accuracy plus breakdowns by edge and size.
- `matrix_cases.csv`: one compact row per edge/size case with median/p95 errors.

The validator checks bundle integrity and completeness, but deliberately does not
encode an accuracy threshold. Accuracy is a reported experimental result, not a
condition that decides whether evidence is kept.
