from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path

from magellan.runtime.dendro import DendroCheckpointDiscovery
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)


class CheckpointValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointSummary:
    task_id: str
    size_bytes: int
    file_count: int
    manifest_path: Path | None
    checkpoint_step: int | None = None


def directory_size_bytes(directory: Path) -> int:
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    )


class CheckpointManager:
    """
    Validates generic single-file checkpoints or manifest-based
    multi-file checkpoints.
    """

    def __init__(
        self,
        registry: PersistentTaskRegistry,
        dendro_discovery: DendroCheckpointDiscovery | None = None,
    ) -> None:
        self._registry = registry
        self._dendro_discovery = (
            dendro_discovery or DendroCheckpointDiscovery()
        )
        self._digest_cache: dict[
            tuple[str, int, int, str], bool
        ] = {}

    def validate(
        self,
        task_id: str,
    ) -> CheckpointSummary:
        checkpoint_directory = (
            self._registry.checkpoint_directory(task_id)
        )

        if not checkpoint_directory.is_dir():
            raise CheckpointValidationError(
                f"Checkpoint directory does not exist: "
                f"{checkpoint_directory}"
            )

        manifest_path = (
            self._registry.checkpoint_manifest_file(task_id)
        )
        definition = self._registry.get_definition(task_id)
        options = definition.runtime.dendro_options
        discovery_spec = (
            options.checkpoint_discovery
            if options is not None
            else None
        )
        if (
            definition.runtime.adapter == "dendro"
            and manifest_path is not None
            and discovery_spec is not None
        ):
            self._dendro_discovery.discover(
                checkpoint_directory=checkpoint_directory,
                manifest_path=manifest_path,
                spec=discovery_spec,
            )

        if manifest_path is None:
            checkpoint_file = (
                self._registry.checkpoint_file(task_id)
            )

            if not checkpoint_file.is_file():
                raise CheckpointValidationError(
                    f"Checkpoint file does not exist: "
                    f"{checkpoint_file}"
                )

            return CheckpointSummary(
                task_id=task_id,
                size_bytes=directory_size_bytes(
                    checkpoint_directory
                ),
                file_count=sum(
                    path.is_file()
                    for path in checkpoint_directory.rglob("*")
                ),
                manifest_path=None,
            )

        if not manifest_path.is_file():
            raise CheckpointValidationError(
                f"Checkpoint manifest does not exist: "
                f"{manifest_path}"
            )

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise CheckpointValidationError(
                f"Invalid checkpoint manifest: {exc}"
            ) from exc

        files = manifest.get("files")

        if not isinstance(files, list) or not files:
            raise CheckpointValidationError(
                "Checkpoint manifest contains no files"
            )

        for item in files:
            if not isinstance(item, dict):
                raise CheckpointValidationError(
                    "Invalid checkpoint manifest entry"
                )

            relative_text = item.get("path")
            expected_size = item.get("size_bytes")

            if not isinstance(relative_text, str):
                raise CheckpointValidationError(
                    "Manifest file path is invalid"
                )

            relative_path = Path(relative_text)

            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
            ):
                raise CheckpointValidationError(
                    f"Unsafe manifest path: {relative_text}"
                )

            file_path = checkpoint_directory / relative_path

            if not file_path.is_file():
                raise CheckpointValidationError(
                    f"Manifest file is missing: {file_path}"
                )

            actual_size = file_path.stat().st_size

            if (
                not isinstance(expected_size, int)
                or actual_size != expected_size
            ):
                raise CheckpointValidationError(
                    f"Checkpoint size mismatch for {file_path}: "
                    f"expected={expected_size}, "
                    f"actual={actual_size}"
                )

            expected_digest = item.get("sha256")
            if expected_digest is not None:
                stat = file_path.stat()
                cache_key = (
                    str(file_path),
                    stat.st_mtime_ns,
                    stat.st_size,
                    str(expected_digest),
                )
                if cache_key not in self._digest_cache:
                    checksum = sha256()
                    with file_path.open("rb") as handle:
                        for chunk in iter(
                            lambda: handle.read(1024 * 1024),
                            b"",
                        ):
                            checksum.update(chunk)
                    self._digest_cache[cache_key] = (
                        checksum.hexdigest() == expected_digest
                    )
                if not self._digest_cache[cache_key]:
                    raise CheckpointValidationError(
                        f"Checkpoint digest mismatch for {file_path}"
                    )

        checkpoint_step = manifest.get("checkpoint_step")
        if checkpoint_step is not None and not isinstance(checkpoint_step, int):
            raise CheckpointValidationError(
                "checkpoint_step must be an integer"
            )

        return CheckpointSummary(
            task_id=task_id,
            size_bytes=directory_size_bytes(
                checkpoint_directory
            ),
            file_count=len(files),
            manifest_path=manifest_path,
            checkpoint_step=checkpoint_step,
        )
