# Stage 4 all-node Dendro environment parity

Stage 4 treats all seven GCP workers as valid geographic destinations for
Dendro/BSSN workloads.  Runtime provisioning is deliberately performed before
measurement so package installation and build transfer are not counted as task
migration overhead.

The final experiment environment requires every configured node to have:

- OpenMPI 4.1.4 (`mpirun` and `mpiexec`),
- the same `~/dgr-build/BSSN_GR/bssnSolver` bytes,
- the same `~/q1-magellan-magellan.toml` template,
- no unresolved dynamic libraries for `bssnSolver`, and
- successful local two-rank MPI process launch.

`scripts/bootstrap_gcp_node.sh` installs/verifies OpenMPI on every worker.
After bootstrap, run `scripts/provision_dendro_all_gcp_nodes.py` from the Boston
worker to mirror the known-good Dendro build and parameter template across the
private seven-node network.  The provisioner validates SHA-256 identity and MPI
spawn on every node before printing `SEVEN_NODE_DENDRO_PROVISION_PASS`.

The Stage 3C workload population generator consequently treats every cluster
node as Dendro-eligible by default.  `--dendro-nodes` remains available for an
experiment that intentionally restricts placement.
