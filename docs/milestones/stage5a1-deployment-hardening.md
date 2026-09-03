# Stage 5A.1 — Effective deployment environment hardening

## Motivation

Stage 5A proved exact code/data identity and a complete seven-node peer mesh, but a
later Stage 5D diagnostic exposed an environment-hygiene gap: the base
`magellan.service` unit specified `runtime-state-gcp` while an older experiment-mode
systemd drop-in still made the *effective* daemon environment use
`runtime-state-gcp-measurement` and `MAGELLAN_CARBON_METRIC=lifecycle`.

The Stage 5A-D runs remain valid because all seven daemons were synchronized and the
real distributed mechanisms worked, but Stage 5E must start from an unambiguous
runtime environment. Stage 5A.1 makes the effective systemd configuration part of
the deployment evidence.

## Canonical Stage 5 environment

Stage 5 deploy mode now owns the Magellan service mode and requires:

- `MAGELLAN_CONFIG=config/cluster.gcp.json`
- `MAGELLAN_POLICY=config/policy.prod.json`
- `MAGELLAN_DATASETS=datasets`
- `MAGELLAN_CARBON_METRIC=lifecycle`
- `MAGELLAN_STATE_ROOT=<repo>/runtime-state-gcp`
- `MAGELLAN_REMOTE_STATE_ROOT=<repo>/runtime-state-gcp`
- `MAGELLAN_REPOSITORY_ROOT=<repo>`
- the exact node ID, Git SHA, and Git branch being deployed.

Lifecycle carbon remains explicit because the calibrated Stage 4 evaluation and
measured Stage 5 experiments use lifecycle carbon. This does not change the daemon's
general backwards-compatible default when it is installed outside the Stage 5A
experiment deployment path.

## Drop-in cleanup

Stage 5A deploy mode invokes the systemd installer with
`MAGELLAN_CLEAR_SYSTEMD_DROPINS=1`. The installer removes Magellan-specific drop-in
directories in `/etc/systemd/system` and `/run/systemd/system` before installing and
restarting the canonical unit. This prevents historical smoke/measurement overrides
from silently changing Stage 5 execution.

The post-restart probe reads `systemctl show ... Environment` and `DropInPaths`; PASS
requires an exact effective environment and zero remaining drop-ins on all seven
nodes.

## Safe state-root transition

The existing measured runs use `runtime-state-gcp-measurement`. Stage 5A.1 does not
delete that directory.

When the effective daemon root differs from the intended `runtime-state-gcp` root,
deployment first requires the current daemon to have zero active tasks. If an old,
inactive `runtime-state-gcp` directory already contains state, it is renamed to a
timestamped `runtime-state-gcp.pre-stage5a1-*` archive. A fresh production root is
then created.

This avoids both failure modes:

1. deleting evidence from the previous measurement root; and
2. accidentally resurrecting stale development registry/catalog state from an old
   production root.

On later Stage 5A deployments, once the daemon already uses the canonical root, the
root is preserved normally across restarts.

## Hardened PASS criteria

In addition to the original Stage 5A checks, a format-v2 deployment bundle requires:

- 7/7 exact effective systemd environments;
- 7/7 nodes with zero Magellan service drop-ins;
- lifecycle carbon active in `/health` on 7/7 nodes;
- writable local and remote state roots on 7/7 nodes;
- the effective state roots to equal `<repo>/runtime-state-gcp`;
- the effective config, policy, datasets, repository root, node ID, Git SHA, and
  branch to match the deployment inputs.

Old Stage 5A format-v1 bundles remain validator-compatible; the new invariants apply
to format-v2 bundles generated after this hardening patch.
