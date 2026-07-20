from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import inf

from magellan.bidding.models import (
    AuctionStrategy,
    BidRecord,
)
from magellan.bidding.resources import dominant_resource_share
from magellan.config.models import NodeResourceCapacity
from magellan.config.policy_models import AuctionPolicy


@dataclass(frozen=True)
class RankedBid:
    bid: BidRecord
    sort_key: tuple
    ranking_value: float
    metrics: dict[str, float | int | str | bool | None]


def _context_value(bid: BidRecord, name: str, default):
    if bid.task_context is None:
        return default
    value = getattr(bid.task_context, name)
    return default if value is None else value


def _deadline_metrics(
    bid: BidRecord,
    now_utc: datetime,
    policy: AuctionPolicy,
) -> tuple[float, float | None]:
    deadline = _context_value(bid, "deadline_at_utc", None)
    remaining = _context_value(
        bid,
        "estimated_remaining_seconds",
        0.0,
    )

    if deadline is None:
        return 0.0, None

    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    wall_slack = (deadline - now_utc).total_seconds()
    completion_slack = wall_slack - remaining
    window = policy.deadline_urgency_window_seconds
    urgency = max(0.0, min(2.0, 1.0 - completion_slack / window))
    return urgency, completion_slack


def rank_bids(
    bids: list[BidRecord],
    strategy: AuctionStrategy,
    credits: dict[str, float],
    node_resources: NodeResourceCapacity,
    policy: AuctionPolicy,
    now_utc: datetime,
) -> list[RankedBid]:
    ranked: list[RankedBid] = []

    for bid in bids:
        remaining = _context_value(
            bid,
            "estimated_remaining_seconds",
            None,
        )
        credit = credits.get(bid.task_id, 0.0)
        opportunity_loss = float(
            _context_value(bid, "opportunity_loss", 0.0)
        )
        priority = int(_context_value(bid, "priority", 0))
        urgency, deadline_slack = _deadline_metrics(
            bid,
            now_utc,
            policy,
        )
        request = (
            bid.task_context.resource_request
            if bid.task_context is not None
            else None
        )
        dominant_share = (
            dominant_resource_share(request, node_resources)
            if request is not None
            else 1.0
        )
        migration_value = max(
            opportunity_loss,
            max(0.0, 1.0 - bid.candidate.score),
        )
        efficiency = migration_value / max(
            dominant_share,
            policy.resource_efficiency_floor,
        )
        urgency_value = urgency + priority / 100.0

        metrics: dict[str, float | int | str | bool | None] = {
            "candidate_score": bid.candidate.score,
            "estimated_remaining_seconds": remaining,
            "credit": credit,
            "opportunity_loss": opportunity_loss,
            "priority": priority,
            "deadline_completion_slack_seconds": deadline_slack,
            "deadline_urgency": urgency,
            "dominant_resource_share": dominant_share,
            "resource_efficiency": efficiency,
        }

        tie = (bid.received_at_utc, bid.bid_id)

        if strategy == AuctionStrategy.LOWEST_SCORE:
            key = (bid.candidate.score, *tie)
            ranking_value = bid.candidate.score
        elif strategy == AuctionStrategy.SHORTEST_REMAINING:
            value = inf if remaining is None else float(remaining)
            key = (value, bid.candidate.score, *tie)
            ranking_value = value
        elif strategy == AuctionStrategy.LONGEST_REMAINING:
            value = -1.0 if remaining is None else float(remaining)
            key = (-value, bid.candidate.score, *tie)
            ranking_value = value
        elif strategy == AuctionStrategy.CREDIT_FAIR:
            key = (-credit, bid.candidate.score, *tie)
            ranking_value = credit
        elif strategy == AuctionStrategy.HIGHEST_REGRET:
            key = (-opportunity_loss, bid.candidate.score, *tie)
            ranking_value = opportunity_loss
        elif strategy == AuctionStrategy.PRIORITY_DEADLINE:
            key = (-urgency_value, bid.candidate.score, *tie)
            ranking_value = urgency_value
        elif strategy == AuctionStrategy.RESOURCE_EFFICIENCY:
            key = (-efficiency, bid.candidate.score, *tie)
            ranking_value = efficiency
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"Unsupported auction strategy: {strategy}")

        ranked.append(
            RankedBid(
                bid=bid,
                sort_key=key,
                ranking_value=ranking_value,
                metrics=metrics,
            )
        )

    return sorted(ranked, key=lambda item: item.sort_key)
