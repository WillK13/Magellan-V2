# Stage 5C — Real destination-side measured-resource contention

## Question

When several real source daemons simultaneously want the same scarce
destination, does the destination daemon itself enforce measured resource
capacity and admit only the bids that physically fit its configured resource
ledger?

Stage 5B establishes real multi-origin decentralized scheduling and successful
peer-to-peer migrations, but intentionally uses lightweight 0.1-core task
requests so it does not create scarcity. Stage 5C isolates the missing
destination-contention behavior.

## Real deployment

Stage 5C runs only after a fresh Stage 5A bundle has verified that all seven
GCP daemons are running the exact Stage 5C Git SHA.

Real source daemons:

- Boston
- California
- South Australia
- Virginia

Real destination daemon:

- Ethiopia

Boston coordinates timing only. It does not rank bids, reserve destination
resources, choose winners, or force migration actions.

## Measured resource shape

The experiment loads the frozen Stage 4D.1
`benchmark-json-medium` resource request rather than introducing a synthetic
slot:

- CPU: approximately 0.9972 cores
- memory: 13 MB
- GPU: 0

Ethiopia's production cluster configuration exposes 2 CPU cores.

One task with the frozen benchmark request is kept resident at Ethiopia before
the challengers are evaluated.

Therefore:

- resident + one challenger ≈ 1.9944 cores: fits;
- resident + two challengers ≈ 2.9917 cores: does not fit.

This creates exactly one additional measured admission.

The runtime process is still the application-checkpoint counter workload. The
benchmark resource vector is deliberately a declared admission request, not a
claim that the counter physically consumes one CPU core. Physical co-location
of the real benchmark/LLM/Dendro workloads is Stage 5E.

## Auction behavior

All four source evaluations are triggered concurrently at the controlled Stage
5B summer trace timestamp. Based on the already-observed Stage 5B production
decisions, each source should independently select Ethiopia.

Ethiopia's normal `BidArbiter` then:

1. collects the real peer bids in its configured auction window;
2. ranks them with the production `lowest_score` policy;
3. builds its `ResourceLedger` from the resident task and active reservations;
4. accepts the highest-ranked request that fits;
5. rejects the other individually feasible requests because remaining CPU is
   insufficient;
6. reserves resources for the accepted migration;
7. marks successful activation as `consumed`.

## PASS criteria

PASS requires:

- four successful source-daemon evaluation triggers;
- four production scheduler decisions recorded on the correct source daemons;
- all four decisions select migration to Ethiopia;
- four real challenger bids stored by Ethiopia;
- exactly one successful bid outcome (`accepted` or final `consumed`);
- exactly three `rejected` outcomes;
- all three rejections explicitly caused by destination resource contention;
- exactly one successful real migration and zero migration failures;
- the resident task remains owned by Ethiopia;
- exactly one challenger becomes owned by Ethiopia;
- the other three challengers remain at their source owners;
- ownership converges across all seven nodes.

No winner identity or score ordering is a PASS criterion.

## Outputs

- `sources.csv`
- `decisions.csv`
- `bids.csv`
- `migrations.csv`
- `ownership.csv`
- `final_tasks.csv`
- `auction_before.json`
- `auction_after.json`
- `events.jsonl`
- `node_evidence.jsonl`
- `metadata.json`
- `summary.json`
- `checksums.sha256`
