# Magellan V2 task-to-location auction

Magellan's market direction is **task to destination**. A destination never bids for work.

1. A running task evaluates `CONTINUE`, `PAUSE`, and destination-specific `MIGRATE` actions.
2. If it wants France, its current owner submits a task bid to France.
3. France gathers all task bids arriving during one auction window.
4. France filters bids that cannot fit its available task slots, CPU cores, memory, GPUs, or accelerator types.
5. France ranks the feasible task bids using its configured auction policy.
6. Accepted bids become expiring resource reservations; ownership changes only after activation commits.
7. Rejected tasks continue at their current owners and may bid again later.

If 30 tasks all want France, those 30 tasks compete in France's local auction. France does not choose tasks globally and does not advertise a reverse bid.

## Bid context

Every task bid contains the normal Magellan migration candidate plus:

- estimated remaining compute time, `Tc`;
- task priority and optional deadline;
- checkpoint and missing static-artifact sizes;
- accumulated cost and cost cap;
- CPU, memory, GPU, and accelerator requirements;
- the task's best fallback action if this bid is rejected;
- the fallback score;
- opportunity loss: `fallback_score - requested_migration_score`.

A large opportunity loss means this task suffers much more than another task if it cannot claim the destination.

## Auction policies

Set the policy in `policy.*.json` under `auction.strategy`, or override it for one daemon with `MAGELLAN_AUCTION_STRATEGY`.

### `lowest_score`

The base Magellan policy. The smallest normalized time/carbon/cost migration score wins.

### `shortest_remaining`

The task with the smallest estimated remaining compute time `Tc` wins. The Magellan score breaks ties.

### `longest_remaining`

The task with the largest `Tc` wins. This can favor jobs that have the largest window over which to amortize migration costs and carbon savings.

### `credit_fair`

A feasible task that loses because capacity is claimed by another task earns destination-specific credit. Higher-credit tasks rank first in later windows. An accepted task spends or decays its credit according to policy.

Credits are persisted in `control/bids.json`, so fairness survives daemon restarts. Impossible resource requests do not earn credit.

### `highest_regret`

The task with the largest opportunity loss wins. This implements the policy: if Task B's fallback is dramatically worse while Task A has an almost-equivalent second choice, prioritize Task B.

### `priority_deadline`

Ranks tasks using configured priority plus urgency derived from deadline slack and estimated remaining work. Tasks likely to miss their deadlines rise above comfortable tasks.

### `resource_efficiency`

Ranks by migration value per dominant resource share. It can admit several smaller high-value tasks instead of one task that consumes all available cores, memory, or GPUs.

## Resource admission

Ranking never bypasses hard capacity checks. The arbiter subtracts:

- resources used by tasks already owned by the destination;
- resources held by accepted or activating reservations;
- task-count slots already owned or reserved.

It then greedily considers ranked tasks and reserves only bids that fit the remaining capacity. Accelerator type is checked for GPU bids.

## Invariants

- `bidder_type` is always `task`.
- Each bid names exactly one desired destination.
- Accepted bids are leases, not ownership.
- A destination cannot reserve more configured resources than it owns.
- Generation-fenced ownership changes only after activation succeeds.
- Credit cannot make an incompatible task feasible.
