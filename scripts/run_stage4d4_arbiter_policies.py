#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from magellan.config.loader import load_policy_config
from magellan.experiments.bundle import (
    validate_checksums,
    write_checksums,
    write_csv,
    write_json,
)
from magellan.experiments.stage4b import (
    CORE_WORKLOADS,
    load_workload_calibrations,
)
from magellan.experiments.stage4d2 import read_resource_model
from magellan.experiments.stage4d4 import (
    BENCHMARK_CLASS,
    STRATEGY_VALUES,
    required_strategies,
    run_fixed_cohort,
    run_starvation_stream,
    verify_single_measured_slot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 4D.4 controlled measured-capacity arbiter-policy evaluation."
    )
    parser.add_argument("--stage4d3-bundle", required=True)
    parser.add_argument("--policy", default="config/policy.prod.json")
    parser.add_argument("--measurements-root", default="experiments/measurements")
    parser.add_argument("--comparison-id")
    parser.add_argument("--destination-node", default="ethiopia")
    return parser.parse_args()


def require_bundle(path: Path, label: str) -> dict:
    errors = validate_checksums(path)
    if errors:
        raise RuntimeError(f"{label} checksum validation failed: " + "; ".join(errors))
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("passed") is not True:
        raise RuntimeError(f"{label} summary passed=false")
    return summary


def main() -> int:
    args = parse_args()
    d43 = Path(args.stage4d3_bundle)
    d43_summary = require_bundle(d43, "Stage 4D.3")

    d42 = Path(str(d43_summary.get("source_stage4d2_bundle") or ""))
    d42_summary = require_bundle(d42, "Stage 4D.2")
    d41 = Path(str(d42_summary.get("source_stage4d1_bundle") or ""))
    require_bundle(d41, "Stage 4D.1")

    a2 = Path(str(d42_summary.get("stage4a2_bundle") or ""))
    a3 = Path(str(d42_summary.get("stage4a3_bundle") or ""))
    a4 = Path(str(d42_summary.get("stage4a4_bundle") or ""))
    require_bundle(a2, "Stage 4A.2")
    require_bundle(a3, "Stage 4A.3")
    require_bundle(a4, "Stage 4A.4")

    capacities, requests = read_resource_model(d41)
    if args.destination_node not in capacities:
        raise RuntimeError(f"Unknown Stage 4D.1 destination node {args.destination_node}")
    benchmark_request = requests[BENCHMARK_CLASS]
    capacity = capacities[args.destination_node]

    calibrations = load_workload_calibrations(
        stage4a2_bundle=a2,
        stage4a3_bundle=a3,
        stage4a4_bundle=a4,
        class_ids=CORE_WORKLOADS,
    )
    benchmark_calibration = calibrations[BENCHMARK_CLASS]
    target_seconds = float(d42_summary.get("target_boston_runtime_seconds") or 0.0)
    if target_seconds <= 0:
        raise RuntimeError("Stage 4D.2 target runtime is missing")

    policy = load_policy_config(args.policy)
    strategies = required_strategies()
    slot_check = verify_single_measured_slot(
        capacity=capacity,
        benchmark_request=benchmark_request,
    )

    comparison_id = args.comparison_id or (
        f"stage4d4-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid4().hex[:8]}"
    )
    root = Path(args.measurements_root) / comparison_id
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)

    print("== Stage 4D.4 measured-capacity arbiter policy evaluation ==")
    print(f"comparison_id={comparison_id}")
    print(f"source_stage4d3={d43}")
    print(f"source_stage4d2={d42}")
    print(f"source_stage4d1={d41}")
    print(f"destination={args.destination_node}")
    print(
        "contention=one measured benchmark remains resident; residual resources "
        "fit exactly one additional measured benchmark"
    )
    print(
        f"benchmark_request=cpu:{benchmark_request.cpu_cores:.6f} "
        f"memory:{benchmark_request.memory_mb}MB"
    )
    print(
        f"node_capacity=cpu:{capacity.cpu_cores:.3f} "
        f"memory:{capacity.memory_mb}MB"
    )
    print(f"strategies={','.join(STRATEGY_VALUES)}")
    print("experiments=fixed_cohort,starvation_stream")

    all_events = []
    fixed_summary = []
    starvation_summary = []

    for value in STRATEGY_VALUES:
        strategy = strategies[value]
        fixed_rows, fixed = run_fixed_cohort(
            strategy=strategy,
            capacity=capacity,
            benchmark_request=benchmark_request,
            calibration=benchmark_calibration,
            policy=policy,
            target_seconds=target_seconds,
            source_node_id="boston",
            destination_node_id=args.destination_node,
        )
        all_events.extend(fixed_rows)
        fixed_summary.append(fixed)
        print(
            f"[fixed] {value:20s} first={fixed['first_winner']:8s} "
            f"order={fixed['admission_order']} "
            f"max_wait={fixed['max_wait_rounds']}"
        )

        stream_rows, stream = run_starvation_stream(
            strategy=strategy,
            capacity=capacity,
            benchmark_request=benchmark_request,
            calibration=benchmark_calibration,
            policy=policy,
            target_seconds=target_seconds,
            source_node_id="boston",
            destination_node_id=args.destination_node,
        )
        all_events.extend(stream_rows)
        starvation_summary.append(stream)
        admitted = stream["persistent_task_admitted"]
        admission_round = stream["persistent_admission_round"]
        print(
            f"[stream] {value:20s} persistent_admitted={admitted} "
            f"round={admission_round} "
            f"rejections={stream['persistent_rejections']} "
            f"final_credit={stream['persistent_final_credit']:.3f}"
        )

    passed = (
        len(fixed_summary) == len(STRATEGY_VALUES)
        and len(starvation_summary) == len(STRATEGY_VALUES)
        and all(row["all_tasks_admitted"] for row in fixed_summary)
        and slot_check["first_fits"] is True
        and slot_check["second_fits"] is False
    )

    summary = {
        "comparison_id": comparison_id,
        "passed": passed,
        "source_stage4d3_bundle": str(d43),
        "source_stage4d2_bundle": str(d42),
        "source_stage4d1_bundle": str(d41),
        "stage4a2_bundle": str(a2),
        "stage4a3_bundle": str(a3),
        "stage4a4_bundle": str(a4),
        "destination_node_id": args.destination_node,
        "strategy_values": list(STRATEGY_VALUES),
        "benchmark_resource_request": {
            "cpu_cores": benchmark_request.cpu_cores,
            "memory_mb": benchmark_request.memory_mb,
            "gpu_count": benchmark_request.gpu_count,
        },
        "destination_capacity": {
            "cpu_cores": capacity.cpu_cores,
            "memory_mb": capacity.memory_mb,
            "gpu_count": capacity.gpu_count,
        },
        "background_resident_benchmark_count": 1,
        "residual_measured_benchmark_admissions": 1,
        "fixed_cohort_bidder_count": 5,
        "starvation_stream_max_rounds": 32,
        "event_count": len(all_events),
    }
    metadata = {
        "format_version": 1,
        "measurement_type": "stage4d4_controlled_arbiter_policy_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "purpose": (
                "Isolate destination-arbiter policy behavior under true feasible contention. "
                "Unlike Stage 4D.2/4D.3, a rejection here is not caused by the bidder being "
                "individually infeasible."
            ),
            "resource_grounding": (
                "Destination capacity and benchmark CPU/memory requests come directly from the "
                "frozen Stage 4D.1 measured resource model. One measured benchmark remains resident, "
                "leaving residual resources for exactly one additional measured benchmark."
            ),
            "measured_task_attributes": (
                "Checkpoint bytes and effective power come from the frozen Stage 4A.2/4A.3/4A.4 "
                "benchmark calibration inherited through Stage 4D.2."
            ),
            "controlled_attributes": (
                "Candidate scores, remaining-work fractions, and opportunity-loss values are "
                "controlled orthogonal inputs. They are not presented as production carbon scores; "
                "they intentionally isolate lowest-score, shortest/longest remaining, regret, and "
                "credit-fair ranking semantics."
            ),
            "fixed_cohort": (
                "Five bidders compete for one measured residual admission. After each winner's "
                "reservation is released, the remaining bidders compete again until all are admitted. "
                "This yields admission order and wait-round metrics."
            ),
            "starvation_stream": (
                "One persistent worse-score bidder competes against one fresh better-score bidder each "
                "round. Rejected bidders receive the production configured credit increment. The test "
                "records whether and when each strategy admits the persistent task."
            ),
            "pass_condition": (
                "PASS validates strategy coverage, measured single-admission contention, deterministic "
                "completion of the fixed cohort, and bundle integrity. PASS is deliberately not tied to "
                "a preferred policy winning."
            ),
        },
    }

    write_csv(
        root / "auction_events.csv",
        all_events,
        list(all_events[0].keys()),
    )
    write_csv(
        root / "fixed_cohort_summary.csv",
        fixed_summary,
        list(fixed_summary[0].keys()),
    )
    write_csv(
        root / "starvation_summary.csv",
        starvation_summary,
        list(starvation_summary[0].keys()),
    )
    write_json(root / "metadata.json", metadata)
    write_json(root / "summary.json", summary)
    write_checksums(root)

    marker = "STAGE_4D4_ARBITER_POLICY_PASS" if passed else "STAGE_4D4_ARBITER_POLICY_FAIL"
    print(f"\n{marker}")
    print(f"bundle: {root}")
    print(f"strategies: {len(fixed_summary)}/{len(STRATEGY_VALUES)}")
    print(f"events: {len(all_events)}")
    print("measured_residual_admissions: 1")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
