# Experiment Infrastructure Stage 3A: Real-system measurement

Stage 3A validates the physical and model inputs used by Magellan before any NSDI
policy-comparison campaign is run. It is measurement infrastructure, not a headline
scheduler comparison.

## Network characterization

`measure_seven_node_network.py` runs from a cluster node with private reachability and
measures all 42 directed source/destination paths. It records repeated HTTP RTT
samples and repeated incompressible rsync-over-SSH transfers using the same `-az`
transport mode as checkpoint migration. Each edge summary preserves the currently
active Magellan telemetry estimate and compares its predicted transfer duration with
the measured duration. Raw samples, summary CSVs, metadata, and SHA-256 checksums are
written to an immutable measurement bundle.

## Controlled migration timing

`measure_migration_model.py` creates a checkpointable counter with an incompressible
payload inside its checkpoint directory. An operator-triggered migration is used only
to force a requested measurement path; the normal Magellan scoring and destination
bid/reservation path is still exercised. The script records the exact migrate
candidate produced by Magellan immediately before migration and compares its predicted
checkpoint, transfer, restore, and total downtime terms with the structured
`migration_completed` event.

The synthetic payload already exists when migration begins. Therefore the measured
`checkpoint_seconds` in Stage 3A represents process quiescence/stop plus checkpoint
validation, not serialization of a new application checkpoint. Stage 3B repeats the
important measurements with real LLM and Dendro checkpoint behavior.

## Isolation

Measurement mode uses the production seven-node topology and 900-second scheduler
epoch, but switches all daemons to `runtime-state-gcp-measurement` and explicitly sets
life-cycle carbon accounting. Entering measurement mode clears this state by default;
`--preserve-measurement-state` keeps accumulated telemetry/calibration when a campaign
is intentionally continued.
