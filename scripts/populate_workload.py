#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magellan.config.loader import load_cluster_config  # noqa: E402
from magellan.experiments.workload_population import (  # noqa: E402
    generate_population,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
    parse_weighted_mix,
    write_population_plan,
)


def read_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic heterogeneous Magellan workload population "
            "and optionally submit/start it across the cluster"
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--mix",
        default="nbody=1,json=1,matmul=1",
        help=(
            "Comma-separated workload weights. Supported: "
            "nbody,json,matmul,dendro,llm"
        ),
    )
    parser.add_argument("--population-id", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--benchmark-iterations", type=int, default=1000)
    parser.add_argument("--mean-interarrival-seconds", type=float, default=0.0)
    parser.add_argument("--initial-nodes", default=None)

    parser.add_argument(
        "--dendro-definition",
        default="config/submissions/dendro-bssn-template.json",
    )
    parser.add_argument("--dendro-solver", default=None)
    parser.add_argument("--dendro-parameter-template", default=None)
    parser.add_argument("--dendro-resolutions", default="8,9,10")
    parser.add_argument("--dendro-time-ends", default="0.5,1.0,2.0")
    parser.add_argument(
        "--dendro-nodes",
        default=None,
        help="Comma-separated eligible Dendro nodes; default is every cluster node",
    )

    parser.add_argument("--llm-definition", default=None)
    parser.add_argument("--llm-nodes", default="boston,virginia")

    parser.add_argument(
        "--submit",
        action="store_true",
        help="Register definitions/runs on their selected initial-owner APIs",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start each submitted task at its scheduled arrival time",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start and not args.submit:
        raise SystemExit("--start requires --submit")

    cluster = load_cluster_config(args.cluster)
    mix = parse_weighted_mix(args.mix)
    population_id = args.population_id or f"population-seed-{args.seed}"
    output = Path(args.output or f"experiments/populations/{population_id}")

    dendro_template = read_json(args.dendro_definition) if "dendro" in mix else None
    llm_template = read_json(args.llm_definition) if "llm" in mix else None
    initial_nodes = (
        parse_csv_strings(args.initial_nodes)
        if args.initial_nodes is not None
        else None
    )

    tasks = generate_population(
        cluster=cluster,
        count=args.count,
        seed=args.seed,
        mix=mix,
        population_id=population_id,
        benchmark_iterations=args.benchmark_iterations,
        mean_interarrival_seconds=args.mean_interarrival_seconds,
        initial_nodes=initial_nodes,
        dendro_template=dendro_template,
        dendro_solver=args.dendro_solver,
        dendro_parameter_template=args.dendro_parameter_template,
        dendro_resolutions=parse_csv_ints(args.dendro_resolutions),
        dendro_time_ends=parse_csv_floats(args.dendro_time_ends),
        dendro_nodes=(
            parse_csv_strings(args.dendro_nodes)
            if args.dendro_nodes is not None
            else None
        ),
        llm_template=llm_template,
        llm_nodes=parse_csv_strings(args.llm_nodes),
    )
    manifest_path = write_population_plan(
        output_directory=output,
        population_id=population_id,
        seed=args.seed,
        mix=mix,
        tasks=tasks,
    )
    print(f"population_manifest={manifest_path}")
    print(f"task_count={len(tasks)}")

    if not args.submit:
        print("WORKLOAD_POPULATION_PLAN_READY")
        return 0

    nodes = {node.id: node for node in cluster.nodes}
    start_wall = time.monotonic()
    submitted: list[dict[str, Any]] = []
    for task in tasks:
        if args.start:
            target = start_wall + task.scheduled_offset_seconds
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        node = nodes[task.initial_node_id]
        api = f"http://{node.internal_ip}:{cluster.api_port}"
        definition_record = post_json(f"{api}/task-definitions", task.definition)
        run_payload = dict(task.run)
        run_payload["revision"] = definition_record["revision"]
        run_payload["auto_start"] = bool(args.start)
        run_view = post_json(f"{api}/task-runs", run_payload)
        submitted.append(
            {
                "index": task.index,
                "node_id": task.initial_node_id,
                "definition_id": definition_record["definition_id"],
                "revision": definition_record["revision"],
                "run_id": run_view["run"]["run_id"],
                "status": run_view["state"]["status"],
            }
        )
        print(
            f"[population] index={task.index} workload={task.workload} "
            f"node={task.initial_node_id} run={run_view['run']['run_id']} "
            f"status={run_view['state']['status']}",
            flush=True,
        )

    submission_path = output / "submitted.json"
    submission_path.write_text(
        json.dumps(submitted, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"submitted_manifest={submission_path}")
    print("WORKLOAD_POPULATION_SUBMITTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
