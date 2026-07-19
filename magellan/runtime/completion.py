from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from magellan.state.persistent_registry import PersistentTaskRegistry


class CompletionValidationError(RuntimeError):
    pass


class CompletionMarker(BaseModel):
    format_version: int = 1
    task_id: str = Field(min_length=1)
    success: bool
    completed_at_utc: datetime
    details: dict = Field(default_factory=dict)


class FinalOutputFile(BaseModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class FinalOutputManifest(BaseModel):
    format_version: int = 1
    task_id: str
    owner_node_id: str
    generation: int = Field(ge=0)
    completed_at_utc: datetime
    created_at_utc: datetime
    total_size_bytes: int = Field(ge=0)
    files: list[FinalOutputFile]
    completion_details: dict = Field(default_factory=dict)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class CompletionManager:
    def __init__(
        self,
        registry: PersistentTaskRegistry,
    ) -> None:
        self._registry = registry

    def completion_marker_exists(self, task_id: str) -> bool:
        path = self._registry.completion_file(task_id)
        return path is not None and path.is_file()

    def load_marker(self, task_id: str) -> CompletionMarker:
        path = self._registry.completion_file(task_id)

        if path is None:
            raise CompletionValidationError(
                f"Task {task_id} does not define a completion marker"
            )

        if not path.is_file():
            raise CompletionValidationError(
                f"Completion marker does not exist: {path}"
            )

        try:
            marker = CompletionMarker.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise CompletionValidationError(
                f"Invalid completion marker {path}: {exc}"
            ) from exc

        if marker.task_id != task_id:
            raise CompletionValidationError(
                f"Completion marker task mismatch: "
                f"expected={task_id}, actual={marker.task_id}"
            )

        if not marker.success:
            raise CompletionValidationError(
                f"Task {task_id} reported unsuccessful completion"
            )

        return marker

    def finalize(
        self,
        task_id: str,
        exit_code: int | None,
    ) -> FinalOutputManifest:
        marker = self.load_marker(task_id)
        state = self._registry.get_state(task_id)
        output_directory = self._registry.output_directory(task_id)
        files: list[FinalOutputFile] = []

        if output_directory is not None:
            if not output_directory.is_dir():
                raise CompletionValidationError(
                    f"Final output directory does not exist: "
                    f"{output_directory}"
                )

            for path in sorted(output_directory.rglob("*")):
                if not path.is_file():
                    continue

                relative = path.relative_to(
                    output_directory
                ).as_posix()

                files.append(
                    FinalOutputFile(
                        path=relative,
                        size_bytes=path.stat().st_size,
                        sha256=sha256_file(path),
                    )
                )

        manifest = FinalOutputManifest(
            task_id=task_id,
            owner_node_id=state.owner_node_id,
            generation=state.generation,
            completed_at_utc=marker.completed_at_utc,
            created_at_utc=datetime.now(timezone.utc),
            total_size_bytes=sum(
                item.size_bytes for item in files
            ),
            files=files,
            completion_details=marker.details,
        )

        manifest_path = (
            self._registry.final_output_manifest_file(task_id)
        )
        manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)

        manifest_sha256 = sha256_file(manifest_path)
        relative_manifest = manifest_path.relative_to(
            self._registry.task_directory(task_id)
        ).as_posix()

        self._registry.mark_completed(
            task_id=task_id,
            completed_at_utc=marker.completed_at_utc,
            manifest_relative_path=relative_manifest,
            manifest_sha256=manifest_sha256,
            output_bytes=manifest.total_size_bytes,
            exit_code=exit_code,
        )

        return manifest

    def load_manifest(
        self,
        task_id: str,
    ) -> FinalOutputManifest:
        path = self._registry.final_output_manifest_file(task_id)

        if not path.is_file():
            raise FileNotFoundError(
                f"Final output manifest does not exist: {path}"
            )

        return FinalOutputManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def resolve_output_file(
        self,
        task_id: str,
        relative_path: str,
    ) -> Path:
        output_directory = self._registry.output_directory(task_id)

        if output_directory is None:
            raise FileNotFoundError(
                f"Task {task_id} has no final output directory"
            )

        requested = Path(relative_path)

        if requested.is_absolute() or ".." in requested.parts:
            raise CompletionValidationError(
                "Output path must be a safe relative path"
            )

        candidate = (output_directory / requested).resolve()
        root = output_directory.resolve()

        if candidate != root and root not in candidate.parents:
            raise CompletionValidationError(
                "Output path escapes the task output directory"
            )

        if not candidate.is_file():
            raise FileNotFoundError(
                f"Final output file does not exist: {candidate}"
            )

        return candidate
