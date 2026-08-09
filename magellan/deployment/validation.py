from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from magellan.carbon.store import CARBON_COLUMN, TIME_COLUMN, as_utc_timestamp
from magellan.config.models import ClusterConfig
from magellan.config.policy_models import ScoringPolicy


@dataclass(frozen=True)
class DatasetSummary:
    node_id: str
    dataset_file: str
    row_count: int
    start_utc: str
    end_utc: str
    median_interval_seconds: float | None
    maximum_gap_seconds: float | None
    sha256: str


@dataclass
class DeploymentValidationReport:
    node_ids: list[str]
    dataset_summaries: list[DatasetSummary] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    common_start_utc: str | None = None
    common_end_utc: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seconds(value: pd.Timedelta) -> float:
    return float(value.total_seconds())


def _validate_resource_consistency(
    cluster: ClusterConfig,
    report: DeploymentValidationReport,
) -> None:
    gce_name = re.compile(r"^[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$")
    for node in cluster.nodes:
        if not gce_name.fullmatch(node.vm_name):
            report.errors.append(
                f"{node.id}: vm_name={node.vm_name!r} is not a valid GCE "
                "instance name; refresh cluster.gcp.json from live GCP metadata"
            )

        resources = node.resources
        capabilities = node.capabilities

        pairs = [
            ("cpu_cores", resources.cpu_cores, capabilities.cpu_cores),
            ("memory_mb", resources.memory_mb, capabilities.memory_mb),
            ("gpu_count", resources.gpu_count, capabilities.gpu_count),
        ]
        for name, configured, capability in pairs:
            if (
                configured is not None
                and capability is not None
                and configured != capability
            ):
                report.errors.append(
                    f"{node.id}: resources.{name}={configured} does not "
                    f"match capabilities.{name}={capability}"
                )

        if resources.accelerator_types != capabilities.accelerator_types:
            report.errors.append(
                f"{node.id}: resource and capability accelerator types differ"
            )

        required_commands = {"bash", "python3", "rsync"}
        missing_commands = required_commands - capabilities.commands
        if missing_commands:
            report.errors.append(
                f"{node.id}: configured capabilities are missing commands "
                f"{sorted(missing_commands)}"
            )

        required_features = {
            "local-command",
            "python-module",
            "process-group",
            "application-checkpoint",
        }
        missing_features = required_features - capabilities.features
        if missing_features:
            report.errors.append(
                f"{node.id}: configured capabilities are missing features "
                f"{sorted(missing_features)}"
            )


def _load_dataset_summary(
    *,
    node_id: str,
    dataset_file: str,
    datasets_directory: Path,
    report: DeploymentValidationReport,
) -> tuple[DatasetSummary | None, pd.Timestamp | None, pd.Timestamp | None]:
    path = datasets_directory / dataset_file
    if not path.is_file():
        report.errors.append(f"{node_id}: missing carbon dataset {path}")
        return None, None, None

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        report.errors.append(f"{node_id}: cannot read {path}: {exc}")
        return None, None, None

    required = {TIME_COLUMN, CARBON_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        report.errors.append(
            f"{node_id}: {dataset_file} is missing columns {sorted(missing)}"
        )
        return None, None, None

    try:
        timestamps = pd.to_datetime(frame[TIME_COLUMN], utc=True, errors="raise")
        values = pd.to_numeric(frame[CARBON_COLUMN], errors="raise")
    except Exception as exc:
        report.errors.append(
            f"{node_id}: invalid carbon dataset values in {dataset_file}: {exc}"
        )
        return None, None, None

    if len(frame) == 0:
        report.errors.append(f"{node_id}: {dataset_file} contains no rows")
        return None, None, None

    if values.isna().any():
        report.errors.append(f"{node_id}: {dataset_file} contains missing carbon values")
    if (values < 0).any():
        report.errors.append(f"{node_id}: {dataset_file} contains negative carbon values")

    index = pd.DatetimeIndex(timestamps).sort_values()
    duplicate_count = int(index.duplicated().sum())
    if duplicate_count:
        report.warnings.append(
            f"{node_id}: {dataset_file} contains {duplicate_count} duplicate timestamps; "
            "CarbonStore will average duplicates"
        )

    unique_index = index.drop_duplicates()
    intervals = unique_index.to_series().diff().dropna()
    median_interval = (
        float(intervals.median().total_seconds()) if not intervals.empty else None
    )
    maximum_gap = (
        float(intervals.max().total_seconds()) if not intervals.empty else None
    )
    if (
        median_interval is not None
        and maximum_gap is not None
        and maximum_gap > 2.5 * median_interval
    ):
        report.warnings.append(
            f"{node_id}: {dataset_file} has a maximum timestamp gap of "
            f"{maximum_gap:.0f}s versus median {median_interval:.0f}s"
        )

    start = unique_index[0]
    end = unique_index[-1]
    summary = DatasetSummary(
        node_id=node_id,
        dataset_file=dataset_file,
        row_count=len(frame),
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        median_interval_seconds=median_interval,
        maximum_gap_seconds=maximum_gap,
        sha256=_sha256(path),
    )
    return summary, start, end


def validate_deployment(
    *,
    cluster: ClusterConfig,
    policy: ScoringPolicy,
    datasets_directory: str | Path,
    expected_node_ids: set[str] | None = None,
) -> DeploymentValidationReport:
    datasets_path = Path(datasets_directory)
    node_ids = [node.id for node in cluster.nodes]
    report = DeploymentValidationReport(node_ids=node_ids)

    if expected_node_ids is not None:
        actual = set(node_ids)
        if actual != expected_node_ids:
            report.errors.append(
                "Cluster node set differs from expected seven-node deployment: "
                f"missing={sorted(expected_node_ids - actual)}, "
                f"unexpected={sorted(actual - expected_node_ids)}"
            )

    if len(cluster.nodes) < 2:
        report.errors.append("Deployment must contain at least two nodes")

    _validate_resource_consistency(cluster, report)

    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    dataset_to_nodes: dict[str, list[str]] = {}

    for node in cluster.nodes:
        dataset_to_nodes.setdefault(node.dataset_file, []).append(node.id)
        summary, start, end = _load_dataset_summary(
            node_id=node.id,
            dataset_file=node.dataset_file,
            datasets_directory=datasets_path,
            report=report,
        )
        if summary is not None:
            report.dataset_summaries.append(summary)
        if start is not None and end is not None:
            starts.append(start)
            ends.append(end)

    for dataset_file, mapped_nodes in dataset_to_nodes.items():
        if len(mapped_nodes) > 1:
            report.warnings.append(
                f"{dataset_file} is shared by nodes {sorted(mapped_nodes)}"
            )

    if len(starts) == len(cluster.nodes) and len(ends) == len(cluster.nodes):
        common_start = max(starts)
        common_end = min(ends)
        report.common_start_utc = common_start.isoformat()
        report.common_end_utc = common_end.isoformat()

        if common_start > common_end:
            report.errors.append("Carbon datasets have no common timestamp overlap")
        elif policy.clock.mode == "trace":
            if policy.clock.trace_start_utc is None:
                report.errors.append("Trace clock requires trace_start_utc")
            else:
                trace_start = as_utc_timestamp(policy.clock.trace_start_utc)
                if not common_start <= trace_start <= common_end:
                    report.errors.append(
                        f"trace_start_utc={trace_start.isoformat()} is outside common "
                        f"dataset range [{common_start.isoformat()}, {common_end.isoformat()}]"
                    )
                else:
                    required_end = trace_start + pd.Timedelta(
                        seconds=max(
                            policy.horizon_seconds,
                            policy.carbon_forecast.horizon_seconds,
                            max(policy.pause.idle_candidates(), default=0.0),
                        )
                    )
                    if required_end > common_end:
                        report.errors.append(
                            "Common dataset range is too short for the configured "
                            f"forecast/scoring horizon: need through {required_end.isoformat()}"
                        )

                    history_seconds = (
                        (policy.carbon_forecast.history_points - 1)
                        * policy.carbon_forecast.sample_interval_seconds
                    )
                    history_start = trace_start - pd.Timedelta(seconds=history_seconds)
                    if history_start < common_start:
                        report.warnings.append(
                            "Trace starts before a full linear-forecast history window is "
                            "available; early forecasts will use persistence/fallback until "
                            "enough causal history accumulates"
                        )

    return report
