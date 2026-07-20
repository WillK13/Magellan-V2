from __future__ import annotations

import re

from magellan.capabilities.models import (
    CompatibilityResult,
    NodeRuntimeCapabilities,
    TaskCompatibilityRequirements,
)


def _version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", value)
    return tuple(int(item) for item in numbers[:4]) or (0,)


def _version_matches(actual: str, specification: str) -> bool:
    """Support a small deterministic subset of PEP-440-style constraints."""
    actual_tuple = _version_tuple(actual)
    for raw_clause in specification.split(","):
        clause = raw_clause.strip()
        if not clause:
            continue
        operator = "=="
        expected_text = clause
        for candidate in (">=", "<=", "==", ">", "<"):
            if clause.startswith(candidate):
                operator = candidate
                expected_text = clause[len(candidate):].strip()
                break
        expected = _version_tuple(expected_text)
        if operator == ">=" and not actual_tuple >= expected:
            return False
        if operator == "<=" and not actual_tuple <= expected:
            return False
        if operator == ">" and not actual_tuple > expected:
            return False
        if operator == "<" and not actual_tuple < expected:
            return False
        if operator == "==" and not actual_tuple[: len(expected)] == expected:
            return False
    return True


def check_compatibility(
    requirements: TaskCompatibilityRequirements,
    capabilities: NodeRuntimeCapabilities,
) -> CompatibilityResult:
    reasons: list[str] = []

    if requirements.architectures:
        if capabilities.architecture not in requirements.architectures:
            reasons.append(
                "task requires architecture in "
                f"{sorted(requirements.architectures)}; destination advertises "
                f"{capabilities.architecture or 'unknown'}"
            )

    if requirements.operating_systems:
        if capabilities.operating_system not in requirements.operating_systems:
            reasons.append(
                "task requires operating system in "
                f"{sorted(requirements.operating_systems)}; destination advertises "
                f"{capabilities.operating_system or 'unknown'}"
            )

    if (
        requirements.minimum_cpu_cores is not None
        and (
            capabilities.cpu_cores is None
            or capabilities.cpu_cores < requirements.minimum_cpu_cores
        )
    ):
        reasons.append(
            f"task requires at least {requirements.minimum_cpu_cores} CPU cores; "
            f"destination advertises {capabilities.cpu_cores}"
        )

    if (
        requirements.minimum_memory_mb is not None
        and (
            capabilities.memory_mb is None
            or capabilities.memory_mb < requirements.minimum_memory_mb
        )
    ):
        reasons.append(
            f"task requires at least {requirements.minimum_memory_mb} MB memory; "
            f"destination advertises {capabilities.memory_mb}"
        )

    if (
        requirements.minimum_gpu_count is not None
        and (
            capabilities.gpu_count is None
            or capabilities.gpu_count < requirements.minimum_gpu_count
        )
    ):
        reasons.append(
            f"task requires at least {requirements.minimum_gpu_count} GPUs; "
            f"destination advertises {capabilities.gpu_count}"
        )

    if requirements.accelerator_types and not (
        requirements.accelerator_types & capabilities.accelerator_types
    ):
        reasons.append(
            "task requires an accelerator in "
            f"{sorted(requirements.accelerator_types)}; destination advertises "
            f"{sorted(capabilities.accelerator_types)}"
        )

    missing_commands = requirements.required_commands - capabilities.commands
    if missing_commands:
        reasons.append(
            f"destination is missing commands {sorted(missing_commands)}"
        )

    missing_features = requirements.required_features - capabilities.features
    if missing_features:
        reasons.append(
            f"destination is missing features {sorted(missing_features)}"
        )

    for runtime, specification in requirements.required_runtimes.items():
        actual = capabilities.runtimes.get(runtime)
        if actual is None:
            reasons.append(
                f"destination does not advertise runtime {runtime}"
            )
        elif not _version_matches(actual, specification):
            reasons.append(
                f"task requires {runtime}{specification}; destination advertises "
                f"{actual}"
            )

    return CompatibilityResult(
        compatible=not reasons,
        reasons=reasons,
        checked_requirements=requirements.model_dump(mode="json"),
        advertised_capabilities=capabilities.model_dump(mode="json"),
    )
