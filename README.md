# Magellan V2

Decentralized carbon-aware scheduling and stateful checkpoint migration.

Current milestone: tasks compete for destination capacity through configurable resource-aware auctions with durable fairness credit, fallback-regret ranking, and CPU/memory/GPU admission control.

- Implementation and GCP validation: `docs/milestones/resource-aware-auction.md`
- Task-to-location auction policies: `docs/design/task-bidding-market.md`
- Dynamic task submission: `docs/milestones/dynamic-task-submission.md`
- Durable reconciliation: `docs/milestones/durable-distributed-reconciliation.md`
- V1-compatible future adaptive weights: `docs/design/adaptive-weights-v2.md`
