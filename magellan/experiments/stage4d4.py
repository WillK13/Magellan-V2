from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from magellan.bidding.models import AuctionStrategy, BidRecord, TaskBidContext
from magellan.bidding.ranking import rank_bids
from magellan.bidding.resources import ResourceLedger, sum_requests
from magellan.config.models import NodeResourceCapacity
from magellan.config.policy_models import ScoringPolicy
from magellan.experiments.stage4b import WorkloadCalibration
from magellan.models.types import ActionType, ScoredAction, TaskResourceRequest


BENCHMARK_CLASS = "benchmark-json-medium"

STRATEGY_VALUES = (
    "lowest_score",
    "shortest_remaining",
    "longest_remaining",
    "credit_fair",
    "highest_regret",
)

# Controlled orthogonal attributes. Resource demand, checkpoint size and power are
# measured; these ranking attributes are intentionally controlled so each policy
# can be isolated without carbon-trace or placement confounding.
COHORT_SPECS = (
    # task_id, candidate_score, remaining_fraction, opportunity_loss
    ("task-a", 0.10, 0.60, 0.05),
    ("task-b", 0.20, 0.20, 0.10),
    ("task-c", 0.30, 0.50, 0.90),
    ("task-d", 0.40, 0.35, 0.30),
    ("task-e", 0.50, 0.90, 0.70),
)


@dataclass(frozen=True)
class ControlledBidSpec:
    task_id: str
    candidate_score: float
    remaining_seconds: float
    opportunity_loss: float


def available_strategies() -> dict[str, AuctionStrategy]:
    return {strategy.value: strategy for strategy in AuctionStrategy}


def required_strategies() -> dict[str, AuctionStrategy]:
    available = available_strategies()
    missing = [value for value in STRATEGY_VALUES if value not in available]
    if missing:
        raise RuntimeError(
            "Installed AuctionStrategy enum is missing Stage 4D.4 strategies: "
            + ", ".join(missing)
        )
    return {value: available[value] for value in STRATEGY_VALUES}


def cohort_specs(target_seconds: float) -> list[ControlledBidSpec]:
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    return [
        ControlledBidSpec(
            task_id=task_id,
            candidate_score=score,
            remaining_seconds=target_seconds * remaining_fraction,
            opportunity_loss=opportunity_loss,
        )
        for task_id, score, remaining_fraction, opportunity_loss in COHORT_SPECS
    ]


def residual_capacity_with_background(
    *,
    capacity: NodeResourceCapacity,
    benchmark_request: TaskResourceRequest,
) -> ResourceLedger:
    # One frozen Stage-4D.1 benchmark remains resident. With the measured request
    # (~0.997 core) on a 2-core node this leaves room for exactly one additional
    # benchmark, creating real feasible contention without a synthetic slot cap.
    return ResourceLedger.from_capacity(
        capacity,
        used=sum_requests([benchmark_request]),
    )


def verify_single_measured_slot(
    *,
    capacity: NodeResourceCapacity,
    benchmark_request: TaskResourceRequest,
) -> dict[str, Any]:
    ledger = residual_capacity_with_background(
        capacity=capacity,
        benchmark_request=benchmark_request,
    )
    first_fits, first_reason = ledger.compatible(benchmark_request)
    if first_fits:
        ledger.consume(benchmark_request)
    second_fits, second_reason = ledger.compatible(benchmark_request)
    if not first_fits or second_fits:
        raise RuntimeError(
            "Stage 4D.4 requires exactly one residual measured benchmark admission: "
            f"first_fits={first_fits} second_fits={second_fits}"
        )
    return {
        "first_fits": first_fits,
        "first_reason": first_reason,
        "second_fits": second_fits,
        "second_reason": second_reason,
    }


def _candidate(
    *,
    source_node_id: str,
    destination_node_id: str,
    score: float,
) -> ScoredAction:
    return ScoredAction(
        action=ActionType.MIGRATE,
        source_node_id=source_node_id,
        destination_node_id=destination_node_id,
        time_seconds=5401.0,
        carbon_grams=max(0.0, score),
        cost_usd=max(0.0, score),
        normalized_time=score,
        normalized_carbon=score,
        normalized_cost=score,
        score=score,
    )


def _bid(
    *,
    spec: ControlledBidSpec,
    source_node_id: str,
    destination_node_id: str,
    request: TaskResourceRequest,
    calibration: WorkloadCalibration,
    now_utc: datetime,
    round_index: int,
    suffix: str = "",
) -> BidRecord:
    candidate = _candidate(
        source_node_id=source_node_id,
        destination_node_id=destination_node_id,
        score=spec.candidate_score,
    )
    context = TaskBidContext(
        workload_type=calibration.workload or BENCHMARK_CLASS,
        estimated_remaining_seconds=spec.remaining_seconds,
        checkpoint_bytes=calibration.checkpoint_bytes,
        static_data_bytes=0,
        accumulated_cost_usd=0.0,
        resource_request=request,
        fallback_action=ActionType.CONTINUE,
        fallback_destination_node_id=None,
        fallback_score=spec.candidate_score + spec.opportunity_loss,
        opportunity_loss=spec.opportunity_loss,
        effective_power_kw=calibration.power_kw,
        power_source="stage4a3_frozen_profile",
    )
    return BidRecord(
        bid_id=f"stage4d4:{round_index}:{spec.task_id}{suffix}",
        epoch_id=f"stage4d4:{round_index}",
        task_id=f"{spec.task_id}{suffix}",
        task_context=context,
        source_node_id=source_node_id,
        destination_node_id=destination_node_id,
        candidate=candidate,
        submitted_at_utc=now_utc,
        received_at_utc=now_utc,
    )


def _rank(
    *,
    bids: list[BidRecord],
    strategy: AuctionStrategy,
    credits: dict[str, float],
    capacity: NodeResourceCapacity,
    policy: ScoringPolicy,
    now_utc: datetime,
):
    return rank_bids(
        bids=bids,
        strategy=strategy,
        credits=credits,
        node_resources=capacity,
        policy=policy.auction,
        now_utc=now_utc,
    )


def run_fixed_cohort(
    *,
    strategy: AuctionStrategy,
    capacity: NodeResourceCapacity,
    benchmark_request: TaskResourceRequest,
    calibration: WorkloadCalibration,
    policy: ScoringPolicy,
    target_seconds: float,
    source_node_id: str,
    destination_node_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    waiting = {spec.task_id: spec for spec in cohort_specs(target_seconds)}
    credits = {task_id: 0.0 for task_id in waiting}
    rows: list[dict[str, Any]] = []
    admission_order: list[str] = []
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for round_index in range(1, len(waiting) + 1):
        bids = [
            _bid(
                spec=spec,
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
                request=benchmark_request,
                calibration=calibration,
                now_utc=now,
                round_index=round_index,
            )
            for spec in waiting.values()
        ]
        ranked = _rank(
            bids=bids,
            strategy=strategy,
            credits=dict(credits),
            capacity=capacity,
            policy=policy,
            now_utc=now,
        )
        ledger = residual_capacity_with_background(
            capacity=capacity,
            benchmark_request=benchmark_request,
        )

        winner: str | None = None
        for rank, item in enumerate(ranked, start=1):
            bid = item.bid
            task_id = bid.task_id
            before = float(credits.get(task_id, 0.0))
            fits, reason = ledger.compatible(benchmark_request)
            if winner is None and fits:
                ledger.consume(benchmark_request)
                after = before * policy.auction.accepted_credit_decay
                credits[task_id] = after
                status = "accepted"
                winner = task_id
            else:
                after = min(
                    policy.auction.credit_max,
                    before + policy.auction.credit_increment,
                )
                credits[task_id] = after
                status = "rejected"
                reason = reason or "single measured residual benchmark admission already consumed"

            rows.append(
                {
                    "experiment": "fixed_cohort",
                    "strategy": strategy.value,
                    "round_index": round_index,
                    "task_id": task_id,
                    "auction_rank": rank,
                    "status": status,
                    "candidate_score": bid.candidate.score,
                    "remaining_seconds": bid.task_context.estimated_remaining_seconds,
                    "opportunity_loss": bid.task_context.opportunity_loss,
                    "credit_before": before,
                    "credit_after": after,
                    "dominant_resource_share": item.metrics.get("dominant_resource_share"),
                    "resource_efficiency": item.metrics.get("resource_efficiency"),
                    "decision_reason": reason or "",
                }
            )

        if winner is None:
            raise RuntimeError(f"No bidder admitted under {strategy.value} in round {round_index}")
        admission_order.append(winner)
        waiting.pop(winner)
        now += timedelta(minutes=15)

    accepted_rows = [row for row in rows if row["status"] == "accepted"]
    waits = {row["task_id"]: int(row["round_index"]) - 1 for row in accepted_rows}
    summary = {
        "strategy": strategy.value,
        "first_winner": admission_order[0],
        "admission_order": "->".join(admission_order),
        "mean_wait_rounds": sum(waits.values()) / len(waits),
        "max_wait_rounds": max(waits.values()),
        "total_rejections": sum(row["status"] == "rejected" for row in rows),
        "all_tasks_admitted": len(admission_order) == len(COHORT_SPECS),
    }
    return rows, summary


def run_starvation_stream(
    *,
    strategy: AuctionStrategy,
    capacity: NodeResourceCapacity,
    benchmark_request: TaskResourceRequest,
    calibration: WorkloadCalibration,
    policy: ScoringPolicy,
    target_seconds: float,
    source_node_id: str,
    destination_node_id: str,
    max_rounds: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The persistent task is deliberately worse on candidate score than every new
    # challenger. Under a pure score policy, a continuous stream can starve it.
    # Credit-based fairness can promote the repeatedly rejected task if the
    # production credit rule is strong enough. This is a controlled liveness test,
    # not a claim about production arrival distributions.
    persistent = ControlledBidSpec(
        task_id="persistent-old",
        candidate_score=0.60,
        remaining_seconds=target_seconds * 0.60,
        opportunity_loss=0.60,
    )
    credits: dict[str, float] = {persistent.task_id: 0.0}
    rows: list[dict[str, Any]] = []
    admitted_round: int | None = None
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for round_index in range(1, max_rounds + 1):
        challenger = ControlledBidSpec(
            task_id=f"fresh-{round_index:02d}",
            candidate_score=0.10,
            remaining_seconds=target_seconds * 0.30,
            opportunity_loss=0.10,
        )
        credits.setdefault(challenger.task_id, 0.0)
        bids = [
            _bid(
                spec=persistent,
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
                request=benchmark_request,
                calibration=calibration,
                now_utc=now,
                round_index=round_index,
            ),
            _bid(
                spec=challenger,
                source_node_id=source_node_id,
                destination_node_id=destination_node_id,
                request=benchmark_request,
                calibration=calibration,
                now_utc=now,
                round_index=round_index,
            ),
        ]
        ranked = _rank(
            bids=bids,
            strategy=strategy,
            credits=dict(credits),
            capacity=capacity,
            policy=policy,
            now_utc=now,
        )
        ledger = residual_capacity_with_background(
            capacity=capacity,
            benchmark_request=benchmark_request,
        )

        winner: str | None = None
        for rank, item in enumerate(ranked, start=1):
            bid = item.bid
            task_id = bid.task_id
            before = float(credits.get(task_id, 0.0))
            fits, reason = ledger.compatible(benchmark_request)
            if winner is None and fits:
                ledger.consume(benchmark_request)
                after = before * policy.auction.accepted_credit_decay
                credits[task_id] = after
                status = "accepted"
                winner = task_id
            else:
                after = min(
                    policy.auction.credit_max,
                    before + policy.auction.credit_increment,
                )
                credits[task_id] = after
                status = "rejected"
                reason = reason or "single measured residual benchmark admission already consumed"

            rows.append(
                {
                    "experiment": "starvation_stream",
                    "strategy": strategy.value,
                    "round_index": round_index,
                    "task_id": task_id,
                    "auction_rank": rank,
                    "status": status,
                    "candidate_score": bid.candidate.score,
                    "remaining_seconds": bid.task_context.estimated_remaining_seconds,
                    "opportunity_loss": bid.task_context.opportunity_loss,
                    "credit_before": before,
                    "credit_after": after,
                    "dominant_resource_share": item.metrics.get("dominant_resource_share"),
                    "resource_efficiency": item.metrics.get("resource_efficiency"),
                    "decision_reason": reason or "",
                }
            )

        if winner == persistent.task_id:
            admitted_round = round_index
            break
        now += timedelta(minutes=15)

    persistent_rows = [row for row in rows if row["task_id"] == persistent.task_id]
    summary = {
        "strategy": strategy.value,
        "persistent_task_admitted": admitted_round is not None,
        "persistent_admission_round": admitted_round,
        "persistent_rejections": sum(row["status"] == "rejected" for row in persistent_rows),
        "persistent_final_credit": (
            float(persistent_rows[-1]["credit_after"]) if persistent_rows else 0.0
        ),
        "rounds_observed": (
            admitted_round if admitted_round is not None else max_rounds
        ),
    }
    return rows, summary


def acceptance_capacity_check(
    *,
    accepted_count: int,
) -> None:
    if accepted_count != 1:
        raise RuntimeError(
            f"Controlled Stage 4D.4 round must accept exactly one bidder, got {accepted_count}"
        )
