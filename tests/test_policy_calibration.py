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


def test_simplex_grid_generation_is_complete_and_deterministic() -> None:
    from magellan.policy.calibration import generate_simplex_weight_grid

    grid = generate_simplex_weight_grid(0.5)

    assert len(grid.weights) == 6
    assert all(
        abs(item.time + item.carbon + item.cost - 1.0) < 1e-12
        for item in grid.weights
    )
    assert grid.weights[0].model_dump() == {
        "time": 0.0,
        "carbon": 0.0,
        "cost": 1.0,
    }
    assert grid.weights[-1].model_dump() == {
        "time": 1.0,
        "carbon": 0.0,
        "cost": 0.0,
    }


def test_calibration_writes_selected_weights_into_policy(tmp_path) -> None:
    import json

    from magellan.policy.calibration import write_calibrated_policy

    result = select_calibrated_baseline(
        [
            candidate(
                "selected",
                {"time": 0.2, "carbon": 0.6, "cost": 0.2},
                10,
                10,
                10,
            )
        ]
    )
    template = tmp_path / "policy.json"
    output = tmp_path / "calibrated.json"
    template.write_text(
        json.dumps(
            {
                "horizon_seconds": 60,
                "weights": {"time": 1, "carbon": 0, "cost": 0},
            }
        ),
        encoding="utf-8",
    )

    write_calibrated_policy(
        template_path=template,
        output_path=output,
        result=result,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["weights"] == {
        "time": 0.2,
        "carbon": 0.6,
        "cost": 0.2,
    }
    assert payload["calibration"]["selected_label"] == "selected"
