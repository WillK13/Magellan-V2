from __future__ import annotations

from pathlib import Path

from scripts.run_stage5a_seven_node_deployment import deploy_remote_command
from scripts.stage5a_node_probe import (
    expected_effective_environment,
    parse_systemd_environment,
)


def test_parse_systemd_environment_handles_empty_and_quoted_values() -> None:
    parsed = parse_systemd_environment(
        'MAGELLAN_NODE_ID=boston MAGELLAN_TASK_FILES= '
        '"MAGELLAN_STATE_ROOT=/home/WILL/Magellan-V2/runtime-state-gcp"'
    )
    assert parsed["MAGELLAN_NODE_ID"] == "boston"
    assert parsed["MAGELLAN_TASK_FILES"] == ""
    assert parsed["MAGELLAN_STATE_ROOT"].endswith("runtime-state-gcp")


def test_expected_stage5_environment_is_production_root_with_lifecycle() -> None:
    env = expected_effective_environment(
        node_id="boston",
        git_sha="a" * 40,
        git_branch="stage5e-real-workloads",
        repository_root=Path("/home/WILL/Magellan-V2"),
        cluster_path="config/cluster.gcp.json",
        policy_path="config/policy.prod.json",
        datasets_path="datasets",
    )
    assert env["MAGELLAN_STATE_ROOT"] == "/home/WILL/Magellan-V2/runtime-state-gcp"
    assert env["MAGELLAN_REMOTE_STATE_ROOT"] == env["MAGELLAN_STATE_ROOT"]
    assert env["MAGELLAN_CARBON_METRIC"] == "lifecycle"


def test_stage5a_deploy_owns_systemd_mode_and_prepares_state_root() -> None:
    command = deploy_remote_command(
        remote_repo="/home/WILL/Magellan-V2",
        branch="stage5d-seven-node-ring",
        target_sha="a" * 40,
        node_id="california",
        service="magellan",
    )
    assert "MAGELLAN_CLEAR_SYSTEMD_DROPINS=1" in command
    assert "MAGELLAN_PREPARE_STATE_ROOT=1" in command
    assert "MAGELLAN_INSTALL_CARBON_METRIC=lifecycle" in command
    assert ".venv/bin/python -m compileall -q magellan scripts" in command
