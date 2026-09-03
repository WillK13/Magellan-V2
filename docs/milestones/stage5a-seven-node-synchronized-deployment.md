# Stage 5A — Real seven-node synchronized deployment

## Purpose

Stage 5A is the boundary between Magellan's offline/replay evaluation and its
real decentralized-system evaluation.

Stages 4B–4E execute scheduler/replay logic from one experiment process. Stage
5A instead requires the **seven actual GCP VMs** to run the same Magellan daemon
at the same exact Git commit and verifies every directed peer path from the
actual source VM.

Stage 5A does not claim that a migration or auction occurred. Those behaviors
are evaluated in Stage 5B–5F.

## Exact deployment identity

The systemd installer now embeds:

- `MAGELLAN_GIT_SHA`
- `MAGELLAN_GIT_BRANCH`

into the daemon service environment. `/health` exposes the same values.

The Stage 5A runner verifies both:

1. `git rev-parse HEAD` in each node's repository; and
2. the Git SHA reported by the **running daemon**.

Both must equal the single target SHA from Boston.

This prevents a node from having updated files on disk while an old daemon
process remains alive.

## Input identity

Each node independently hashes:

- `config/cluster.gcp.json`;
- `config/policy.prod.json`;
- all seven configured carbon CSVs.

A bundle passes only if every node has the same config/policy hashes and the
same hash for each dataset. Seven datasets across seven nodes produce 49
dataset-manifest records.

## Directed mesh

Stage 5A reuses `validate_seven_node_mesh.py` from Boston. For every ordered
non-self source/destination pair, the check is executed **from the source VM**.

For seven nodes:

```
7 × 6 = 42 directed paths
```

Every path must pass:

- destination FastAPI `/health`, with the expected destination node id;
- source-to-destination SSH, including `rsync` and `python3` availability.

The SSH check is relevant because Magellan's real artifact/checkpoint migration
transport depends on source-to-destination host connectivity in addition to the
peer API.

## Canonical procedure

After Stage 4E.4 is committed and pushed, create a dedicated Stage 5A branch.
From Boston, run:

```bash
python scripts/run_stage5a_seven_node_deployment.py \
  --stage4e3-bundle "$MAGELLAN_STAGE4E3_CANONICAL" \
  --deploy
```

`--deploy` performs two phases:

1. every non-Boston node fetches the deployment branch, refuses tracked dirty
   state, switches exactly to `origin/<branch>`, verifies the target SHA, and
   recompiles the code;
2. all seven nodes install/restart `magellan.service` from that exact checkout.

Untracked runtime-state and measurement directories are not treated as a dirty
working tree and are not deleted.

After deployment the runner probes every node, validates the 42-path mesh, and
writes a checksummed measurement bundle.

Running without `--deploy` is verification-only and does not alter remote Git or
systemd state.

## Bundle

- `nodes.csv` — repository and running-daemon SHA, service state, capability
  readiness, config/policy hashes;
- `dataset_hashes.csv` — 49 node/dataset hash records;
- `directed_mesh.csv` — all 42 API and SSH/rsync path results;
- `node_probes.jsonl` — complete local `/health` and `/capabilities` evidence;
- `metadata.json`;
- `summary.json`;
- `checksums.sha256`.

## PASS

PASS requires:

- seven exact configured node identities;
- seven clean tracked worktrees at one exact Git SHA;
- seven running daemons reporting that same SHA;
- seven `/health` responses with correct local node identity;
- seven `/capabilities` responses with `ready=true`;
- identical cluster config and production policy hashes;
- identical copies of all seven carbon datasets;
- all 42 directed API paths;
- all 42 directed SSH/rsync paths.

No preferred scheduling outcome is part of Stage 5A PASS.
