from __future__ import annotations

import argparse
import json
from pathlib import Path

from magellan.carbon.store import CarbonStore
from magellan.config.loader import (
    load_cluster_config,
    load_policy_config,
)
from magellan.graph.topology import ClusterGraph
from magellan.models.types import TaskProfile
from magellan.scheduler.scoring import evaluate_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cluster",
        default="config/cluster.dev.json",
    )
    parser.add_argument(
        "--policy",
        default="config/policy.dev.json",
    )
    parser.add_argument(
        "--task",
        default="config/tasks/dev-llm.json",
    )
    parser.add_argument(
        "--datasets",
        default="datasets",
    )
    parser.add_argument(
        "--at",
        default="2024-01-01T12:00:00Z",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cluster = load_cluster_config(args.cluster)
    policy = load_policy_config(args.policy)

    task_raw = json.loads(
        Path(args.task).read_text(encoding="utf-8")
    )
    task = TaskProfile.model_validate(task_raw)

    carbon_store = CarbonStore(
        cluster=cluster,
        datasets_directory=args.datasets,
    )
    graph = ClusterGraph(cluster)

    decision = evaluate_task(
        task=task,
        cluster=cluster,
        policy=policy,
        graph=graph,
        carbon_store=carbon_store,
        at_utc=args.at,
    )

    print()
    print(
        f"Task {task.task_id} at node {task.current_node_id} "
        f"for evaluation time {args.at}"
    )
    print()

    for rank, action in enumerate(decision.ranked_actions, start=1):
        destination = (
            action.destination_node_id
            if action.destination_node_id is not None
            else task.current_node_id
        )

        print(
            f"{rank}. {action.action.value:8s} "
            f"destination={destination:15s} "
            f"score={action.score:.6f} "
            f"time={action.time_seconds:.2f}s "
            f"carbon={action.carbon_grams:.6f}g "
            f"cost=${action.cost_usd:.6f}"
        )

    print()
    print(
        f"SELECTED: {decision.selected.action.value} "
        f"destination="
        f"{decision.selected.destination_node_id or task.current_node_id}"
    )
    print(f"REASON:   {decision.reason}")


if __name__ == "__main__":
    main()
