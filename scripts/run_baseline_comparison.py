#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from magellan.carbon.store import CarbonStore
from magellan.config.loader import load_cluster_config, load_policy_config
from magellan.experiments.baseline_suite import run_baseline_suite
from magellan.experiments.bundle import sha256_file, write_checksums, write_csv, write_json
from magellan.experiments.comparison import ComparisonWorkload, outcome_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Magellan's Stage-2 offline baseline/oracle comparison against "
            "the seven carbon traces. This does not contact or modify GCP nodes."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument(
        "--carbon-metric",
        choices=("lifecycle", "direct"),
        default="lifecycle",
        help=(
            "Carbon-intensity series used for accounting and forecasts. "
            "NSDI experiment comparisons default to life-cycle intensity; "
            "use direct for operational-carbon sensitivity runs."
        ),
    )
    parser.add_argument("--start-utc", default="2024-01-02T00:00:00Z")
    parser.add_argument("--duration-seconds", type=float, default=14_400.0)
    parser.add_argument("--power-kw", type=float, default=0.08)
    parser.add_argument("--checkpoint-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--static-data-bytes", type=int, default=0)
    parser.add_argument("--start-node", default="boston")
    parser.add_argument("--cost-cap-usd", type=float, default=None)
    parser.add_argument("--oracle-quantum-seconds", type=float, default=300.0)
    parser.add_argument("--oracle-max-elapsed-multiplier", type=float, default=3.0)
    parser.add_argument("--output-root", default="experiments/comparisons")
    parser.add_argument("--name", default="stage2-baseline-validation")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def dataset_identity(cluster, directory: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for node in cluster.nodes:
        path = directory / node.dataset_file
        result[node.id] = {
            "file": node.dataset_file,
            "sha256": sha256_file(path),
        }
    return result


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    policy = load_policy_config(args.policy)
    datasets = Path(args.datasets)
    carbon_store = CarbonStore(
        cluster,
        datasets,
        carbon_metric=args.carbon_metric,
    )
    start = pd.Timestamp(args.start_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    workload = ComparisonWorkload(
        name=args.name,
        duration_seconds=args.duration_seconds,
        power_kw=args.power_kw,
        checkpoint_bytes=args.checkpoint_bytes,
        static_data_bytes=args.static_data_bytes,
        start_node_id=args.start_node,
        cost_cap_usd=args.cost_cap_usd,
    )

    print("== Stage 2 baseline + oracle comparison ==")
    print(
        f"carbon_metric={carbon_store.carbon_metric.value} "
        f"column={carbon_store.carbon_column}"
    )
    print(f"start={start.isoformat()} duration={workload.duration_seconds:g}s")
    print(
        f"workload power={workload.power_kw:g}kW "
        f"checkpoint={workload.checkpoint_bytes}B start_node={workload.start_node_id}"
    )
    print("NOTE: offline replay only; no GCP nodes are contacted or modified")

    outcomes, metadata = run_baseline_suite(
        cluster=cluster,
        policy=policy,
        carbon_store=carbon_store,
        workload=workload,
        start_utc=start,
        oracle_quantum_seconds=args.oracle_quantum_seconds,
        oracle_max_elapsed_multiplier=args.oracle_max_elapsed_multiplier,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    comparison_id = f"baseline-{timestamp}-{uuid4().hex[:8]}"
    output = Path(args.output_root) / comparison_id
    output.mkdir(parents=True, exist_ok=False)
    trajectories = output / "trajectories"
    trajectories.mkdir()

    manifest = {
        "format_version": 1,
        "comparison_id": comparison_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_head(),
        "cluster_config": {
            "path": args.cluster,
            "sha256": sha256_file(args.cluster),
        },
        "policy_config": {
            "path": args.policy,
            "sha256": sha256_file(args.policy),
        },
        "datasets": dataset_identity(cluster, datasets),
        "carbon_accounting": {
            "metric": carbon_store.carbon_metric.value,
            "column": carbon_store.carbon_column,
            "units": "gCO2eq/kWh",
        },
        "start_utc": start.isoformat(),
        "workload": workload.model_dump(mode="json"),
        "oracle": {
            "quantum_seconds": args.oracle_quantum_seconds,
            "max_elapsed_multiplier": args.oracle_max_elapsed_multiplier,
        },
        "comparison_scope": (
            "Offline policy-model validation. Magellan causal replay uses the "
            "v1.2 scoring implementation and configured edge fallbacks, not live telemetry."
        ),
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "metadata.json", metadata)

    rows = []
    for outcome in outcomes:
        row = outcome_row(outcome)
        row["global_objective"] = metadata["global_objective_values"][outcome.policy]
        rows.append(row)
        write_json(
            trajectories / f"{outcome.policy}.json",
            outcome.model_dump(mode="json"),
        )

    fields = [
        "policy",
        "start_node_id",
        "selected_initial_node_id",
        "final_node_id",
        "completed",
        "makespan_seconds",
        "compute_seconds",
        "paused_idle_seconds",
        "pause_overhead_seconds",
        "migration_seconds",
        "carbon_grams",
        "cost_usd",
        "migrations",
        "pauses",
        "decision_count",
        "owner_path",
        "global_objective",
    ]
    write_csv(output / "results.csv", rows, fields)
    write_json(output / "results.json", [item.model_dump(mode="json") for item in outcomes])
    write_checksums(output)

    print("\npolicy                  carbon(g)    cost($)    makespan(s)  mig  pause  final")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['policy']:<23} "
            f"{row['carbon_grams']:>10.6f}  "
            f"{row['cost_usd']:>9.6f}  "
            f"{row['makespan_seconds']:>11.1f}  "
            f"{row['migrations']:>3}  "
            f"{row['pauses']:>5}  "
            f"{row['final_node_id']}"
        )

    print("\nBASELINE COMPARISON PASSED")
    print(f"comparison_id: {comparison_id}")
    print(f"bundle: {output}")
    print(f"best_static: {next(x.final_node_id for x in outcomes if x.policy == 'best_static')}")
    print(
        "best_at_dispatch: "
        f"{next(x.final_node_id for x in outcomes if x.policy == 'best_at_dispatch')}"
    )
    oracle = next(x for x in outcomes if x.policy == "clairvoyant_oracle")
    print(f"oracle_path: {' -> '.join(oracle.owner_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
