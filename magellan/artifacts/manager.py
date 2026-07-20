from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from threading import RLock
from uuid import uuid4

from magellan.artifacts.models import (
    ArtifactBinding,
    ArtifactFile,
    ArtifactManifest,
    StaticArtifactSpec,
)
from magellan.state.persistent_registry import (
    PersistentTaskRegistry,
)


class ArtifactValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


class ArtifactManager:
    def __init__(
        self,
        registry: PersistentTaskRegistry,
    ) -> None:
        self._registry = registry
        self._cache_root = (
            registry.state_root / "artifact-cache"
        )
        self._incoming_root = (
            registry.state_root / "artifact-incoming"
        )
        self._lock = RLock()

        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._incoming_root.mkdir(parents=True, exist_ok=True)

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    @property
    def incoming_root(self) -> Path:
        return self._incoming_root

    def cache_directory(self, digest: str) -> Path:
        return self._cache_root / digest

    def payload_directory(self, digest: str) -> Path:
        return self.cache_directory(digest) / "payload"

    def manifest_path(self, digest: str) -> Path:
        return self.cache_directory(digest) / "manifest.json"

    def incoming_directory(
        self,
        migration_id: str,
        digest: str,
    ) -> Path:
        return (
            self._incoming_root
            / migration_id
            / digest
        )

    def _build_manifest(
        self,
        specification: StaticArtifactSpec,
    ) -> ArtifactManifest:
        source = Path(
            specification.source_directory
        ).expanduser().resolve()

        if not source.is_dir():
            raise FileNotFoundError(
                f"Artifact source directory does not exist: "
                f"{source}"
            )

        files: list[ArtifactFile] = []

        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file():
                continue

            relative = file_path.relative_to(source).as_posix()

            files.append(
                ArtifactFile(
                    path=relative,
                    size_bytes=file_path.stat().st_size,
                    sha256=sha256_file(file_path),
                )
            )

        if not files:
            raise ArtifactValidationError(
                f"Artifact directory is empty: {source}"
            )

        digest_payload = {
            "artifact_id": specification.artifact_id,
            "kind": specification.kind,
            "files": [
                file.model_dump()
                for file in files
            ],
        }

        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

        return ArtifactManifest(
            artifact_id=specification.artifact_id,
            kind=specification.kind,
            digest=digest,
            size_bytes=sum(
                file.size_bytes
                for file in files
            ),
            files=files,
        )

    def load_manifest(
        self,
        digest: str,
    ) -> ArtifactManifest:
        path = self.manifest_path(digest)

        if not path.is_file():
            raise ArtifactValidationError(
                f"Artifact manifest does not exist: {path}"
            )

        return ArtifactManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def validate_cache(
        self,
        digest: str,
        verify_hashes: bool = True,
    ) -> ArtifactManifest:
        manifest = self.load_manifest(digest)
        payload = self.payload_directory(digest)

        if not payload.is_dir():
            raise ArtifactValidationError(
                f"Artifact payload does not exist: {payload}"
            )

        for item in manifest.files:
            path = payload / item.path

            if not path.is_file():
                raise ArtifactValidationError(
                    f"Artifact file is missing: {path}"
                )

            actual_size = path.stat().st_size

            if actual_size != item.size_bytes:
                raise ArtifactValidationError(
                    f"Artifact size mismatch for {path}: "
                    f"expected={item.size_bytes}, "
                    f"actual={actual_size}"
                )

            if (
                verify_hashes
                and sha256_file(path) != item.sha256
            ):
                raise ArtifactValidationError(
                    f"Artifact digest mismatch: {path}"
                )

        return manifest

    def has_artifact(self, digest: str) -> bool:
        try:
            self.validate_cache(
                digest,
                verify_hashes=False,
            )
            return True
        except ArtifactValidationError:
            return False

    def _import_source(
        self,
        specification: StaticArtifactSpec,
    ) -> ArtifactManifest:
        manifest = self._build_manifest(specification)

        if self.has_artifact(manifest.digest):
            return self.validate_cache(manifest.digest)

        source = Path(
            specification.source_directory
        ).expanduser().resolve()

        temporary = (
            self._cache_root
            / f".tmp-{manifest.digest}-{uuid4()}"
        )
        payload = temporary / "payload"

        try:
            shutil.copytree(source, payload)

            (temporary / "manifest.json").write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )

            # Validate before publishing it into the cache.
            published = self.cache_directory(
                manifest.digest
            )

            if published.exists():
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, published)

            return self.validate_cache(manifest.digest)

        finally:
            if temporary.exists():
                shutil.rmtree(
                    temporary,
                    ignore_errors=True,
                )

    def bindings_for_task(
        self,
        task_id: str,
    ) -> list[ArtifactBinding]:
        definition = self._registry.get_definition(task_id)
        state = self._registry.get_state(task_id)

        bindings: list[ArtifactBinding] = []

        for specification in definition.artifacts:
            digest = state.artifact_digests.get(
                specification.artifact_id
            )

            if digest is None:
                continue

            manifest = self.load_manifest(digest)

            bindings.append(
                ArtifactBinding(
                    artifact_id=specification.artifact_id,
                    digest=digest,
                    size_bytes=manifest.size_bytes,
                )
            )

        return bindings

    def stage_bindings(
        self,
        task_id: str,
        bindings: list[ArtifactBinding],
    ) -> None:
        definition = self._registry.get_definition(task_id)

        specifications = {
            specification.artifact_id: specification
            for specification in definition.artifacts
        }

        artifacts_directory = (
            self._registry.artifacts_directory(task_id)
        )
        artifacts_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        artifact_digests: dict[str, str] = {}

        for binding in bindings:
            specification = specifications.get(
                binding.artifact_id
            )

            if specification is None:
                raise ArtifactValidationError(
                    f"Unknown artifact for task {task_id}: "
                    f"{binding.artifact_id}"
                )

            self.validate_cache(binding.digest)

            target = (
                artifacts_directory
                / specification.target_relative_directory
            )

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)

            temporary_link = target.with_name(
                f".{target.name}.tmp-{uuid4()}"
            )

            temporary_link.symlink_to(
                self.payload_directory(binding.digest),
                target_is_directory=True,
            )

            os.replace(temporary_link, target)

            artifact_digests[
                binding.artifact_id
            ] = binding.digest

        self._registry.set_artifact_digests(
            task_id,
            artifact_digests,
        )

    def ensure_task_artifacts(
        self,
        task_id: str,
    ) -> list[ArtifactBinding]:
        with self._lock:
            definition = self._registry.get_definition(
                task_id
            )
            state = self._registry.get_state(task_id)

            bindings: list[ArtifactBinding] = []

            for specification in definition.artifacts:
                digest = state.artifact_digests.get(
                    specification.artifact_id
                )

                if (
                    digest is not None
                    and self.has_artifact(digest)
                ):
                    manifest = self.validate_cache(
                        digest,
                        verify_hashes=False,
                    )
                else:
                    manifest = self._import_source(
                        specification
                    )

                bindings.append(
                    ArtifactBinding(
                        artifact_id=(
                            specification.artifact_id
                        ),
                        digest=manifest.digest,
                        size_bytes=manifest.size_bytes,
                    )
                )

            self.stage_bindings(task_id, bindings)
            return bindings

    def commit_incoming(
        self,
        migration_id: str,
        digest: str,
    ) -> ArtifactManifest:
        incoming = self.incoming_directory(
            migration_id,
            digest,
        )

        if not incoming.is_dir():
            raise ArtifactValidationError(
                f"Incoming artifact does not exist: "
                f"{incoming}"
            )

        destination = self.cache_directory(digest)

        with self._lock:
            if destination.exists():
                shutil.rmtree(
                    incoming,
                    ignore_errors=True,
                )
                return self.validate_cache(destination.name)

            os.replace(incoming, destination)

            try:
                return self.validate_cache(
                    digest,
                    verify_hashes=True,
                )
            except Exception:
                shutil.rmtree(
                    destination,
                    ignore_errors=True,
                )
                raise
