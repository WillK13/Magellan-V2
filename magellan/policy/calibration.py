from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import tempfile

from pydantic import BaseModel, Field, model_validator

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


class CalibrationGrid(BaseModel):
    step_size: float = Field(gt=0, le=1)
    weights: list[WeightVector]

    @model_validator(mode="after")
    def validate_unique(self) -> "CalibrationGrid":
        keys = {
            (
                round(item.time, 12),
                round(item.carbon, 12),
                round(item.cost, 12),
            )
            for item in self.weights
        }
        if len(keys) != len(self.weights):
            raise ValueError("Calibration grid contains duplicate weights")
        return self


def _step_units(step_size: float) -> int:
    try:
        step = Decimal(str(step_size))
    except InvalidOperation as exc:
        raise ValueError("Invalid calibration step size") from exc
    if step <= 0 or step > 1:
        raise ValueError("Calibration step size must be in (0, 1]")
    units = Decimal("1") / step
    integral = units.to_integral_value()
    if units != integral:
        raise ValueError(
            "Calibration step size must divide 1 exactly, for example "
            "0.5, 0.25, 0.1, or 0.02"
        )
    return int(integral)


def generate_simplex_weight_grid(step_size: float = 0.02) -> CalibrationGrid:
    """Generate every non-negative (time, carbon, cost) vector summing to 1."""
    units = _step_units(step_size)
    weights: list[WeightVector] = []
    for time_units in range(units + 1):
        for carbon_units in range(units - time_units + 1):
            cost_units = units - time_units - carbon_units
            weights.append(
                WeightVector(
                    time=time_units / units,
                    carbon=carbon_units / units,
                    cost=cost_units / units,
                )
            )
    return CalibrationGrid(step_size=step_size, weights=weights)


def _weight_key(weights: WeightVector) -> tuple[float, float, float]:
    normalized = weights.normalized()
    return (
        round(normalized.time, 12),
        round(normalized.carbon, 12),
        round(normalized.cost, 12),
    )


def validate_grid_coverage(
    candidates: list[CalibrationCandidate],
    grid: CalibrationGrid,
) -> None:
    candidate_keys = {_weight_key(item.weights) for item in candidates}
    grid_keys = {_weight_key(item) for item in grid.weights}
    missing = sorted(grid_keys - candidate_keys)
    extra = sorted(candidate_keys - grid_keys)
    if missing or extra:
        raise ValueError(
            "Calibration candidates do not exactly cover the requested grid: "
            f"missing={len(missing)}, extra={len(extra)}"
        )


def select_calibrated_baseline(
    candidates: list[CalibrationCandidate],
    *,
    cost_cap_usd: float | None = None,
    deadline_seconds: float | None = None,
) -> CalibrationResult:
    if not candidates:
        raise ValueError("At least one calibration candidate is required")

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

    ranked = sorted(
        scored,
        key=lambda item: (
            item.score,
            item.candidate.total_carbon_grams,
            item.candidate.total_time_seconds,
            item.candidate.total_cost_usd,
            _weight_key(item.candidate.weights),
            item.candidate.label or "",
        ),
    )
    return CalibrationResult(
        selected=ranked[0],
        feasible_candidate_count=len(feasible),
        rejected_candidate_count=len(candidates) - len(feasible),
        ranked=ranked,
    )


def write_calibrated_policy(
    *,
    template_path: str | Path,
    output_path: str | Path,
    result: CalibrationResult,
) -> Path:
    template = Path(template_path)
    output = Path(output_path)
    raw = json.loads(template.read_text(encoding="utf-8"))
    weights = result.selected.candidate.weights.normalized()
    raw["weights"] = weights.model_dump(mode="json")
    raw.setdefault("calibration", {})
    raw["calibration"].update(
        {
            "selected_label": result.selected.candidate.label,
            "selected_score": result.selected.score,
            "feasible_candidate_count": result.feasible_candidate_count,
            "rejected_candidate_count": result.rejected_candidate_count,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(raw, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output.parent,
        prefix=output.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    temporary.replace(output)
    return output


def _write_json(path: str | Path, value: BaseModel | dict) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a simplex grid, select feasible Magellan baseline "
            "weights, and optionally write a calibrated policy"
        )
    )
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--cost-cap-usd", type=float)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--step-size", type=float, default=0.02)
    parser.add_argument("--generate-grid-output")
    parser.add_argument("--require-grid-coverage", action="store_true")
    parser.add_argument("--policy-template")
    parser.add_argument("--policy-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = generate_simplex_weight_grid(args.step_size)
    if args.generate_grid_output:
        _write_json(args.generate_grid_output, grid)

    if args.input is None:
        if not args.generate_grid_output:
            raise SystemExit(
                "Provide --input or --generate-grid-output"
            )
        return

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    candidates = [
        CalibrationCandidate.model_validate(item)
        for item in raw["candidates"]
    ]
    if args.require_grid_coverage:
        validate_grid_coverage(candidates, grid)

    result = select_calibrated_baseline(
        candidates,
        cost_cap_usd=args.cost_cap_usd,
        deadline_seconds=args.deadline_seconds,
    )
    rendered = json.dumps(
        result.model_dump(mode="json"),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")

    if bool(args.policy_template) != bool(args.policy_output):
        raise SystemExit(
            "--policy-template and --policy-output must be provided together"
        )
    if args.policy_template and args.policy_output:
        write_calibrated_policy(
            template_path=args.policy_template,
            output_path=args.policy_output,
            result=result,
        )


if __name__ == "__main__":
    main()
