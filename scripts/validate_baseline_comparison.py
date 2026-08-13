#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from magellan.experiments.baseline_suite import REQUIRED_BASELINE_POLICIES
from magellan.experiments.bundle import validate_checksums
from magellan.experiments.comparison import PolicyOutcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Stage-2 baseline/oracle comparison bundle."
    )
    parser.add_argument("bundle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.bundle)
    errors = validate_checksums(root)
    for required in ("manifest.json", "metadata.json", "results.json", "results.csv"):
        if not (root / required).is_file():
            errors.append(f"Missing required file: {required}")

    manifest = {}
    if (root / "manifest.json").is_file():
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid manifest.json: {exc}")

    outcomes: list[PolicyOutcome] = []
    if (root / "results.json").is_file():
        try:
            raw = json.loads((root / "results.json").read_text(encoding="utf-8"))
            outcomes = [PolicyOutcome.model_validate(item) for item in raw]
        except Exception as exc:
            errors.append(f"Invalid results.json: {exc}")

    by_policy = {item.policy: item for item in outcomes}
    missing = set(REQUIRED_BASELINE_POLICIES) - set(by_policy)
    if missing:
        errors.append(f"Missing baseline policies: {sorted(missing)}")
    if len(by_policy) != len(outcomes):
        errors.append("Duplicate policy result in results.json")

    expected_compute = manifest.get("workload", {}).get("duration_seconds")
    for item in outcomes:
        if not item.completed:
            errors.append(f"Policy did not complete: {item.policy}")
        if expected_compute is not None and abs(item.compute_seconds - float(expected_compute)) > 1e-6:
            errors.append(f"Unexpected compute duration: {item.policy}")
        trajectory = root / "trajectories" / f"{item.policy}.json"
        if not trajectory.is_file():
            errors.append(f"Missing trajectory: {trajectory.name}")

    if "boston_static" in by_policy and by_policy["boston_static"].final_node_id != "boston":
        errors.append("boston_static did not remain in Boston")
    if "france_static" in by_policy and by_policy["france_static"].final_node_id != "france":
        errors.append("france_static did not remain in France")
    if "temporal_only" in by_policy and by_policy["temporal_only"].migrations != 0:
        errors.append("temporal_only performed a migration")
    if "clairvoyant_oracle" in by_policy:
        oracle = by_policy["clairvoyant_oracle"]
        if oracle.start_node_id != by_policy.get("magellan_causal", oracle).start_node_id:
            errors.append("Oracle and Magellan causal replay use different start nodes")

    if errors:
        print("BASELINE COMPARISON FAILED")
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("BASELINE COMPARISON BUNDLE PASSED")
    print(f"policies: {len(outcomes)}")
    for item in outcomes:
        print(
            f"[OK] {item.policy:<23} carbon={item.carbon_grams:.6f}g "
            f"cost=${item.cost_usd:.6f} makespan={item.makespan_seconds:.1f}s "
            f"path={' -> '.join(item.owner_path)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
