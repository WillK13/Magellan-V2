from __future__ import annotations

from pathlib import Path

import pandas as pd

from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.deployment.validation import validate_deployment


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "boston",
    "california",
    "south-australia",
    "nepal",
    "ethiopia",
    "france",
    "virginia",
}


def _cluster_with_valid_vm_names():
    cluster = load_cluster_config(REPO_ROOT / "config" / "cluster.gcp.json")
    nodes = [
        node.model_copy(update={"vm_name": f"magellan-{node.id}"})
        for node in cluster.nodes
    ]
    return cluster.model_copy(update={"nodes": nodes})


def _write_datasets(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(
        "2024-01-01T00:00:00Z",
        periods=48,
        freq="1h",
    )
    cluster = _cluster_with_valid_vm_names()
    for index, node in enumerate(cluster.nodes):
        frame = pd.DataFrame(
            {
                "Datetime (UTC)": timestamps,
                "Carbon intensity gCO₂eq/kWh (direct)": [
                    50.0 + index + sample for sample in range(len(timestamps))
                ],
            }
        )
        frame.to_csv(directory / node.dataset_file, index=False)


def test_seven_node_deployment_validation_accepts_complete_inputs(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    _write_datasets(datasets)

    report = validate_deployment(
        cluster=_cluster_with_valid_vm_names(),
        policy=load_policy_config(REPO_ROOT / "config" / "policy.prod.json"),
        datasets_directory=datasets,
        expected_node_ids=EXPECTED,
    )

    assert report.valid, report.errors
    assert len(report.dataset_summaries) == 7
    assert report.common_start_utc == "2024-01-01T00:00:00+00:00"
    assert report.common_end_utc == "2024-01-02T23:00:00+00:00"
    assert any("full linear-forecast history" in warning for warning in report.warnings)


def test_seven_node_deployment_validation_reports_missing_dataset(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    _write_datasets(datasets)
    (datasets / "Virginia_24H.csv").unlink()

    report = validate_deployment(
        cluster=_cluster_with_valid_vm_names(),
        policy=load_policy_config(REPO_ROOT / "config" / "policy.prod.json"),
        datasets_directory=datasets,
        expected_node_ids=EXPECTED,
    )

    assert not report.valid
    assert any("Virginia_24H.csv" in error for error in report.errors)


def test_seven_node_deployment_validation_rejects_invalid_gce_name(tmp_path) -> None:
    datasets = tmp_path / "datasets"
    _write_datasets(datasets)
    cluster = _cluster_with_valid_vm_names()
    first = cluster.nodes[0].model_copy(update={"vm_name": "Invalid_Name"})
    cluster = cluster.model_copy(update={"nodes": [first, *cluster.nodes[1:]]})

    report = validate_deployment(
        cluster=cluster,
        policy=load_policy_config(REPO_ROOT / "config" / "policy.prod.json"),
        datasets_directory=datasets,
        expected_node_ids=EXPECTED,
    )

    assert not report.valid
    assert any("not a valid GCE instance name" in error for error in report.errors)
