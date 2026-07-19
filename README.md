# Magellan V2

Decentralized carbon-aware scheduling and stateful checkpoint migration.

Current milestone: durable live telemetry replaces configured task, power, network, and migration-overhead estimates when fresh measurements are available, with automatic stale-data fallback.

- Implementation and GCP validation: `docs/milestones/telemetry-live-models.md`
- Telemetry model contract: `docs/design/telemetry-contract.md`
- Resource-aware task auctions: `docs/milestones/resource-aware-auction.md`
- Task-to-location auction policies: `docs/design/task-bidding-market.md`
- Dynamic task submission: `docs/milestones/dynamic-task-submission.md`
- Durable reconciliation: `docs/milestones/durable-distributed-reconciliation.md`
- V1-compatible future adaptive weights: `docs/design/adaptive-weights-v2.md`
