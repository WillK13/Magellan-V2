from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StaticArtifactSpec(BaseModel):
    artifact_id: str = Field(min_length=1)
    kind: Literal[
        "dataset",
        "model",
        "code",
        "environment",
        "container",
        "other",
    ]

    # Used only when the current peer initially possesses the artifact.
    source_directory: str = Field(min_length=1)

    # Location under:
    # runtime-state/tasks/<task-id>/artifacts/
    target_relative_directory: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> "StaticArtifactSpec":
        from pathlib import Path

        target = Path(self.target_relative_directory)

        if target.is_absolute() or ".." in target.parts:
            raise ValueError(
                "target_relative_directory must be a safe relative path"
            )

        return self


class ArtifactFile(BaseModel):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class ArtifactManifest(BaseModel):
    format_version: int = 1

    artifact_id: str
    kind: str
    digest: str

    size_bytes: int = Field(ge=0)
    files: list[ArtifactFile]


class ArtifactBinding(BaseModel):
    artifact_id: str
    digest: str
    size_bytes: int = Field(ge=0)


class ArtifactStatusRequest(BaseModel):
    task_id: str
    artifacts: list[ArtifactBinding]


class ArtifactStatusResponse(BaseModel):
    task_id: str
    present_digests: list[str]
    missing_digests: list[str]


class ArtifactCommitRequest(BaseModel):
    migration_id: str
    task_id: str
    artifact_id: str
    digest: str


class ArtifactCommitResponse(BaseModel):
    artifact_id: str
    digest: str
    committed: bool
    size_bytes: int = 0
    error: str | None = None
