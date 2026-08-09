# Seven-node GCP deployment

This milestone operationalizes the completed Magellan V2 scheduler across the production seven-node GCP topology. It does not change continue/pause/migrate scoring, adaptive weights, bidding, checkpointing, or recovery semantics.

## Target topology

`config/cluster.gcp.json` defines:

- Boston
- California
- South Australia
- Nepal
- Ethiopia
- France
- Virginia

Every node runs the same FastAPI daemon and must have all seven carbon traces locally because every owner independently evaluates all compatible destinations.

## 1. Prepare carbon datasets

Place these files under `datasets/`:

```text
Boston_24H.csv
California_24H.csv
South_Australia_24H.csv
Nepal_24H.csv
Ethiopia_24H.csv
France_24H.csv
Virginia_24H.csv
```

Then run:

```bash
python scripts/validate_seven_node_deployment.py
```

The validator checks the seven-node identity set, GCE-compatible VM names, resource/capability consistency, required CSV columns, numeric carbon values, common time overlap, trace-clock coverage, forecast horizon coverage, timestamp gaps, and SHA-256 hashes.

CSV files remain ignored by Git. Use:

```bash
python scripts/sync_datasets_to_gcp.py
```

to validate and copy the complete directory to every configured VM.

## 2. Audit live GCP metadata

Before deployment, compare the checked-in topology against live GCP:

```bash
python scripts/audit_gcp_cluster.py
```

The audit matches instances by configured name/zone and falls back to internal IP so stale VM names can be discovered. Correct `config/cluster.gcp.json` until the audit passes.

## 3. Bootstrap every node

Each VM should already contain `~/Magellan-V2`. From a workstation with `gcloud` configured:

```bash
python scripts/bootstrap_all_gcp_nodes.py
```

This pulls `seven-node-deployment`, verifies Python 3.11+, installs the editable environment, and runs the full test suite on every VM. Dataset validation is intentionally skipped during initial bootstrap so datasets can be synchronized immediately afterward.

After dataset sync, rerun:

```bash
python scripts/bootstrap_all_gcp_nodes.py --skip-tests --validate-datasets
```

## 4. Install the daemon as a service

After every node has the repository and datasets:

```bash
python scripts/install_all_gcp_services.py
```

This installs a `magellan.service` systemd unit on every VM. The only per-node difference is `MAGELLAN_NODE_ID`; all nodes use `cluster.gcp.json`, `policy.prod.json`, the same repository path, and `runtime-state-gcp`.

Check a node manually with:

```bash
sudo systemctl status magellan
journalctl -u magellan -f
```

For interactive debugging instead of systemd:

```bash
scripts/start_magellan_node.sh boston
```

## 5. Validate all directed paths

From Boston (or another node specified with `--local-node-id`), verify every source can reach every destination through both the peer API and SSH/rsync:

```bash
python scripts/validate_seven_node_mesh.py \
  --local-node-id boston \
  --ssh-user WILL
```

A seven-node complete directed mesh has 42 non-self paths. The command passes only when all 42 API paths and all 42 SSH/rsync paths work.

## 6. Functional smoke test

After the mesh passes, submit a normal task to Boston and allow the scheduler to operate autonomously. Do not manually force a migration for the primary smoke test. Verify that:

1. all seven peers are visible and reachable;
2. all seven carbon forecasts are available;
3. task telemetry becomes fresh;
4. every scheduler epoch evaluates continue, pause, and feasible migration candidates;
5. if migration wins, the destination auction reserves capacity;
6. static artifacts prefetch while the source keeps running;
7. checkpoint, transfer, restore, ownership, accounting, telemetry, and adaptive policy state move to the destination;
8. the new owner continues making autonomous decisions in later epochs.

The correct scheduler is not required to visit all seven nodes in one run. Route coverage belongs to a separate connectivity/validation test; the autonomous run should be allowed to choose only actions that its live score considers beneficial.

## Deferred from this milestone

- experiment/result generation
- seven-node scaling studies
- SLURM
- JAX/Orbax
- CRIU
- GPU migration
- live ElectricityMaps ingestion
