from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from magellan.models.utils import minmax_normalize
from magellan.policy.models import WeightVector


class CalibrationCandidate(BaseModel):
    weights: WeightVector
    total_time_seconds: float = Field(ge=0)
    total_carbon_grams: float = Field(ge=0)
    total_cost_usd: float = Field(ge=0)
    label: str | None = None


class CalibrationScore(BaseModel):
    candidate: CalibrationCandidate
    normalized_time: float = Field(ge=0, le=1)
    normalized_carbon: float = Field(ge=0, le=1)
    normalized_cost: float = Field(ge=0, le=1)
    score: float = Field(ge=0)


class CalibrationResult(BaseModel):
    selected: CalibrationScore
    feasible_candidate_count: int = Field(ge=1)
    rejected_candidate_count: int = Field(ge=0)
    ranked: list[CalibrationScore]


def select_calibrated_baseline(
    candidates: list[CalibrationCandidate],
    *,
    cost_cap_usd: float | None = None,
    deadline_seconds: float | None = None,
) -> CalibrationResult:
    feasible = [
        candidate
        for candidate in candidates
        if (
            (cost_cap_usd is None or candidate.total_cost_usd <= cost_cap_usd)
            and (
                deadline_seconds is None
                or candidate.total_time_seconds <= deadline_seconds
            )
        )
    ]
    if not feasible:
        raise ValueError("No calibration candidates satisfy hard constraints")

    normalized_time = minmax_normalize(
        [candidate.total_time_seconds for candidate in feasible]
    )
    normalized_carbon = minmax_normalize(
        [candidate.total_carbon_grams for candidate in feasible]
    )
    normalized_cost = minmax_normalize(
        [candidate.total_cost_usd for candidate in feasible]
    )

    scored: list[CalibrationScore] = []
    for index, candidate in enumerate(feasible):
        weights = candidate.weights.normalized()
        score = (
            weights.time * normalized_time[index]
            + weights.carbon * normalized_carbon[index]
            + weights.cost * normalized_cost[index]
        )
        scored.append(
            CalibrationScore(
                candidate=candidate,
                normalized_time=normalized_time[index],
                normalized_carbon=normalized_carbon[index],
                normalized_cost=normalized_cost[index],
                score=score,
            )
        )

    ranked = sorted(scored, key=lambda item: item.score)
    return CalibrationResult(
        selected=ranked[0],
        feasible_candidate_count=len(feasible),
        rejected_candidate_count=len(candidates) - len(feasible),
        ranked=ranked,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select baseline Magellan objective weights from runs"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--cost-cap-usd", type=float)
    parser.add_argument("--deadline-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    candidates = [
        CalibrationCandidate.model_validate(item)
        for item in raw["candidates"]
    ]
    result = select_calibrated_baseline(
        candidates,
        cost_cap_usd=args.cost_cap_usd,
        deadline_seconds=args.deadline_seconds,
    )
    rendered = json.dumps(result.model_dump(mode="json"), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
