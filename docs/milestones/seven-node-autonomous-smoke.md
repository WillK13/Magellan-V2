# Seven-node autonomous smoke validation

This validation is intentionally separate from the production experiment configuration.

- `config/cluster.gcp.json` remains the production seven-node configuration with a 900-second scheduler epoch.
- `config/cluster.gcp.smoke.json` uses the same nodes but a 20-second scheduler epoch.
- `config/policy.gcp.smoke.json` uses a carbon-heavy objective and a rapidly advancing causal trace so a migration is expected quickly without calling an operator migration endpoint.
- Smoke state is isolated under `runtime-state-gcp-smoke`; production state under `runtime-state-gcp` is not deleted or reused.

The validation submits one checkpointable counter task to Boston, waits for the task definition to converge to all peers, and then observes policy state and ownership until natural completion. A pass requires:

1. all seven daemons are using the smoke policy;
2. the definition converges to all seven catalogs;
3. at least one autonomous scheduler decision selects `MIGRATE`;
4. ownership generation changes and reaches the selected destination;
5. the destination subsequently records another autonomous scheduling decision; and
6. the workload naturally completes without an error.

No `/tasks/{task_id}/migrate/...` operator endpoint is called by the smoke validator.

Use `scripts/set_seven_node_service_mode.py smoke` from the operator machine to enter smoke mode, then run `scripts/validate_seven_node_autonomous_smoke.py` from Boston. Return every service to production mode with `scripts/set_seven_node_service_mode.py prod` after validation.
