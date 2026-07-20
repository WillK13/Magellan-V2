from magellan.policy.calibration import (
    CalibrationCandidate,
    select_calibrated_baseline,
)
from magellan.policy.models import WeightVector


def candidate(label, weights, time, carbon, cost):
    return CalibrationCandidate(
        label=label,
        weights=WeightVector(**weights),
        total_time_seconds=time,
        total_carbon_grams=carbon,
        total_cost_usd=cost,
    )


def test_calibration_enforces_hard_constraints_before_ranking() -> None:
    result = select_calibrated_baseline(
        [
            candidate(
                "fast-expensive",
                {"time": 1, "carbon": 0, "cost": 0},
                10,
                100,
                20,
            ),
            candidate(
                "balanced",
                {"time": 0.25, "carbon": 0.5, "cost": 0.25},
                20,
                20,
                5,
            ),
            candidate(
                "slow-cheap",
                {"time": 0, "carbon": 0.5, "cost": 0.5},
                100,
                10,
                1,
            ),
        ],
        cost_cap_usd=10,
        deadline_seconds=50,
    )

    assert result.feasible_candidate_count == 1
    assert result.rejected_candidate_count == 2
    assert result.selected.candidate.label == "balanced"
