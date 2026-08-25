from __future__ import annotations

import csv
import json
from pathlib import Path

from magellan.experiments.bundle import write_checksums
from magellan.experiments.stage4a2 import summarize_profile_samples
from scripts.run_stage4a3_profiles import BENCHMARKS, DENDRO_VARIANTS, SIZES


def test_stage4a3_matrix_has_thirteen_workload_classes() -> None:
    assert len(BENCHMARKS) * len(SIZES) + len(DENDRO_VARIANTS) + 1 == 13
    assert DENDRO_VARIANTS == ((8, 3.0), (9, 1.0), (10, 2.0))


def test_profile_summary_includes_process_count() -> None:
    summary = summarize_profile_samples(
        [
            {"process_count": 3, "cpu_utilization_percent": 150},
            {"process_count": 3, "cpu_utilization_percent": 190},
        ]
    )
    assert summary["process_count"]["median"] == 3
    assert summary["cpu_utilization_percent"]["median"] == 170


def test_profile_only_flags_are_supported() -> None:
    workload = Path("scripts/measure_stage4a2_workload.py").read_text(encoding="utf-8")
    llm = Path("scripts/measure_llm_migration.py").read_text(encoding="utf-8")
    assert '"--profile-only"' in workload
    assert '"--profile-only"' in llm
    assert "STAGE_4A3_PROFILE_MEASUREMENT_PASS" in workload
    assert "STAGE_4A3_PROFILE_MEASUREMENT_PASS" in llm


def test_stage4a3_validator_accepts_minimal_valid_bundle(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "stage4a3-test"
    root.mkdir()
    summary = {
        "calibration_id": "stage4a3-test",
        "node_id": "boston",
        "trials_per_class": 1,
        "expected_class_count": 1,
        "expected_run_count": 1,
        "passed": True,
    }
    case = {
        "case_id": "benchmark-json-small-trial01",
        "passed": True,
        "profile_only": True,
        "profile": {"sample_count": 3},
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "metadata.json").write_text("{}", encoding="utf-8")
    (root / "case_summaries.json").write_text(json.dumps([case]), encoding="utf-8")
    with (root / "profile_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id"])
        writer.writeheader(); writer.writerow({"case_id": case["case_id"]})
    with (root / "profile_classes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "trial_count"])
        writer.writeheader(); writer.writerow({"class_id": "benchmark-json-small", "trial_count": 1})
    write_checksums(root)

    from scripts import validate_stage4a3_profiles as validator
    monkeypatch.setattr("sys.argv", ["validate_stage4a3_profiles.py", str(root)])
    assert validator.main() == 0
