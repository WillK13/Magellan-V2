# Magellan V2

Decentralized carbon-aware scheduling and stateful checkpoint migration.

Current milestone: durable per-task adaptive objective weights and rolling normalization. Historical calibration selects a baseline policy, while budget slack, deadline risk, carbon opportunity, and telemetry confidence adjust bounded effective weights at runtime.

- Adaptive policy implementation and GCP validation: `docs/milestones/adaptive-policy-normalization.md`
- Adaptive policy design contract: `docs/design/adaptive-policy-v2.md`
- Telemetry implementation: `docs/milestones/telemetry-live-models.md`
- Telemetry contract: `docs/design/telemetry-contract.md`
- Resource-aware task auctions: `docs/milestones/resource-aware-auction.md`
- Task-to-location auction policies: `docs/design/task-bidding-market.md`
- Dynamic task submission: `docs/milestones/dynamic-task-submission.md`
- Durable reconciliation: `docs/milestones/durable-distributed-reconciliation.md`
