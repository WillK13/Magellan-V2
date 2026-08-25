# Stage 4A.1 — Final hardware and directed-WAN calibration

Stage 4A.1 is the first measurement milestone after the final Stage-3C experiment-readiness freeze. It does not change scheduling policy. Its purpose is to capture a reproducible physical baseline for every later Stage-4 result.

## Acceptance contract

A valid Stage-4A.1 campaign must:

1. observe the expected seven-node cluster;
2. verify every VM reports the configured `e2-highmem-2` type through GCP metadata;
3. verify CPU/memory capacity, service health, lifecycle carbon mode, measurement-isolated telemetry state, and capability readiness;
4. require the cluster to be idle: no owned/paused tasks, pending bids, active reservations, or reserved resource fraction;
5. require every node to run the exact same Git commit as the launching node;
6. characterize all `N*(N-1)` directed WAN edges (42 for seven nodes);
7. force a fresh transport-faithful affine calibration for each directed edge, then test a separate held-out incompressible rsync/SSH transfer;
8. collect repeated RTT and held-out transfer samples;
9. preserve hardware, raw network samples, descriptive summaries, and SHA-256 checksums in one immutable measurement bundle.

The default Stage-4A.1 campaign uses 15 RTT samples and 3 held-out transfers of 8 MiB + 123 bytes per directed edge. The off-boundary 123-byte suffix guarantees the held-out rsync size cannot equal the 64-KiB-chunked steady-stream calibration sample; these are measurement defaults, not scheduler constants.

## Run

From Boston, with all seven daemons already deployed on the final experiment commit:

```bash
python scripts/run_stage4a1_calibration.py
```

The command writes under `experiments/measurements/stage4a1-*` and finishes with:

```text
STAGE_4A1_CALIBRATION_PASS
```

Validate the resulting bundle independently:

```bash
python scripts/validate_stage4a1_calibration.py \
  experiments/measurements/<stage4a1-id>
```

Expected:

```text
STAGE_4A1_CALIBRATION_BUNDLE_PASS
```

## Output layout

```text
experiments/measurements/<stage4a1-id>/
├── metadata.json
├── hardware.json
├── hardware.csv
├── summary.json
├── checksums.sha256
└── network/
    └── directed-mesh/
        ├── metadata.json
        ├── edges.csv
        ├── rtt_samples.csv
        ├── bandwidth_samples.csv
        └── checksums.sha256
```

Stage 4A.2 must reference the completed Stage-4A.1 bundle rather than silently replacing its WAN/hardware baseline.
