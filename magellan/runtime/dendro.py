from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import time

from magellan.state.persistent_registry import PersistentTaskRegistry
from magellan.runtime.completion import CompletionMarker
from magellan.state.task_models import (
    DendroCheckpointDiscoverySpec,
    DendroCompletionSpec,
    DendroProgressSpec,
)


def _digest(path: Path) -> str:
    checksum = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _extract_number(pattern: str | None, value: str, name: str) -> int | None:
    if pattern is None:
        return None
    match = re.search(pattern, value)
    if match is None:
        return None
    raw = match.groupdict().get(name)
    if raw is None and match.groups():
        raw = match.group(1)
    if raw is None:
        return None
    return int(raw)


class DendroCheckpointDiscovery:
    """Build a complete manifest from stable Dendro rank checkpoints."""

    @staticmethod
    def _write_manifest(
        *,
        manifest_path: Path,
        checkpoint_step: int,
        expected_rank_count: int | None,
        files: list[dict[str, int | str]],
        generation: int | None = None,
    ) -> Path:
        manifest = {
            "format_version": 1,
            "workload_type": "dendro-gr",
            "checkpoint_step": checkpoint_step,
            "expected_rank_count": expected_rank_count,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files,
        }
        if generation is not None:
            manifest["checkpoint_generation"] = generation
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        return manifest_path

    def _discover_native_bssn(
        self,
        *,
        checkpoint_directory: Path,
        manifest_path: Path,
        spec: DendroCheckpointDiscoverySpec,
    ) -> Path | None:
        prefix = spec.native_bssn_prefix
        if prefix is None:
            return None
        now = time.time()
        generation_pattern = re.compile(
            rf"^{re.escape(prefix)}_(?P<generation>\d+)_step\.cp$"
        )
        candidates: list[tuple[int, int, list[tuple[Path, int | None]]]] = []

        for step_path in checkpoint_directory.glob(f"{prefix}_*_step.cp"):
            match = generation_pattern.fullmatch(step_path.name)
            if match is None or not step_path.is_file():
                continue
            generation = int(match.group("generation"))
            try:
                metadata = json.loads(step_path.read_text(encoding="utf-8"))
                checkpoint_step = metadata[spec.native_bssn_step_key]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if not isinstance(checkpoint_step, int):
                continue

            if spec.expected_rank_count is None:
                ranks = sorted(
                    int(match.group("rank"))
                    for path in checkpoint_directory.glob(
                        f"{prefix}_{generation}_*.var"
                    )
                    if (
                        match := re.fullmatch(
                            rf"{re.escape(prefix)}_{generation}_"
                            r"(?P<rank>\d+)\.var",
                            path.name,
                        )
                    )
                )
            else:
                ranks = list(range(spec.expected_rank_count))
            if not ranks:
                continue

            selected: list[tuple[Path, int | None]] = [(step_path, None)]
            complete = True
            for rank in ranks:
                selected.extend(
                    [
                        (
                            checkpoint_directory
                            / f"{prefix}_{generation}_{rank}.var",
                            rank,
                        ),
                        (
                            checkpoint_directory
                            / f"{prefix}_{generation}_octree_{rank}.oct",
                            rank,
                        ),
                    ]
                )
            selected.append(
                (
                    checkpoint_directory
                    / f"{prefix}_aeh_solver_checkpt-cp{generation}.json",
                    None,
                )
            )

            for path, _rank in selected:
                if (
                    not path.is_file()
                    or path.stat().st_size <= 0
                    or now - path.stat().st_mtime < spec.stability_seconds
                ):
                    complete = False
                    break
            if not complete:
                continue
            if (
                spec.expected_file_count is not None
                and len(selected) != spec.expected_file_count
            ):
                continue
            candidates.append((checkpoint_step, generation, selected))

        if not candidates:
            return None
        checkpoint_step, generation, selected = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        selected = sorted(selected, key=lambda item: item[0].name)
        selected_sizes = {
            path.relative_to(checkpoint_directory).as_posix(): path.stat().st_size
            for path, _rank in selected
        }
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_files = existing.get("files", [])
                existing_sizes = {
                    item.get("path"): item.get("size_bytes")
                    for item in existing_files
                    if isinstance(item, dict)
                }
                digests_ready = (
                    not spec.include_sha256
                    or all(
                        isinstance(item, dict) and item.get("sha256")
                        for item in existing_files
                    )
                )
                if (
                    existing.get("checkpoint_step") == checkpoint_step
                    and existing.get("checkpoint_generation") == generation
                    and existing_sizes == selected_sizes
                    and digests_ready
                ):
                    return manifest_path
            except (OSError, json.JSONDecodeError):
                pass

        files: list[dict[str, int | str]] = []
        for path, rank in selected:
            relative = path.relative_to(checkpoint_directory)
            entry: dict[str, int | str] = {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
            }
            if rank is not None:
                entry["rank"] = rank
            if spec.include_sha256:
                entry["sha256"] = _digest(path)
            files.append(entry)
        return self._write_manifest(
            manifest_path=manifest_path,
            checkpoint_step=checkpoint_step,
            expected_rank_count=spec.expected_rank_count,
            files=files,
            generation=generation,
        )

    def discover(
        self,
        *,
        checkpoint_directory: Path,
        manifest_path: Path,
        spec: DendroCheckpointDiscoverySpec,
    ) -> Path | None:
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        if spec.native_bssn_prefix is not None:
            return self._discover_native_bssn(
                checkpoint_directory=checkpoint_directory,
                manifest_path=manifest_path,
                spec=spec,
            )
        now = time.time()
        candidates: dict[Path, tuple[int, int | None]] = {}

        for pattern in spec.file_globs:
            for path in checkpoint_directory.glob(pattern):
                if not path.is_file() or path == manifest_path:
                    continue
                relative = path.relative_to(checkpoint_directory)
                relative_text = relative.as_posix()
                step = _extract_number(spec.step_regex, relative_text, "step")
                if spec.step_regex is not None and step is None:
                    continue
                rank = _extract_number(spec.rank_regex, relative_text, "rank")
                if spec.rank_regex is not None and rank is None:
                    continue
                if now - path.stat().st_mtime < spec.stability_seconds:
                    continue
                candidates[relative] = (step or 0, rank)

        groups: dict[int, list[tuple[Path, int | None]]] = {}
        for relative, (step, rank) in candidates.items():
            groups.setdefault(step, []).append((relative, rank))

        selected_step: int | None = None
        selected: list[tuple[Path, int | None]] = []
        for step in sorted(groups, reverse=True):
            items = groups[step]
            if (
                spec.expected_file_count is not None
                and len(items) != spec.expected_file_count
            ):
                continue
            if spec.expected_rank_count is not None:
                ranks = {rank for _, rank in items if rank is not None}
                if len(ranks) != spec.expected_rank_count:
                    continue
            selected_step = step
            selected = sorted(items, key=lambda item: item[0].as_posix())
            break

        if selected_step is None or not selected:
            return None

        selected_sizes = {
            relative.as_posix(): (
                checkpoint_directory / relative
            ).stat().st_size
            for relative, _rank in selected
        }
        if manifest_path.is_file():
            try:
                existing = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                existing_files = existing.get("files", [])
                existing_sizes = {
                    item.get("path"): item.get("size_bytes")
                    for item in existing_files
                    if isinstance(item, dict)
                }
                digests_ready = (
                    not spec.include_sha256
                    or all(
                        isinstance(item, dict) and item.get("sha256")
                        for item in existing_files
                    )
                )
                if (
                    existing.get("checkpoint_step") == selected_step
                    and existing_sizes == selected_sizes
                    and digests_ready
                ):
                    return manifest_path
            except (OSError, json.JSONDecodeError):
                pass

        files: list[dict[str, int | str]] = []
        for relative, rank in selected:
            path = checkpoint_directory / relative
            entry: dict[str, int | str] = {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
            }
            if rank is not None:
                entry["rank"] = rank
            if spec.include_sha256:
                entry["sha256"] = _digest(path)
            files.append(entry)

        return self._write_manifest(
            manifest_path=manifest_path,
            checkpoint_step=selected_step,
            expected_rank_count=spec.expected_rank_count,
            files=files,
        )


def _read_log_tail(path: Path, maximum_bytes: int) -> str:
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - maximum_bytes))
        payload = handle.read()
    return payload.decode("utf-8", errors="replace")


class DendroCompletionSynthesizer:
    def __init__(self, registry: PersistentTaskRegistry) -> None:
        self._registry = registry

    def synthesize(self, task_id: str, exit_code: int | None) -> bool:
        definition = self._registry.get_definition(task_id)
        options = definition.runtime.dendro_options
        if definition.runtime.adapter != "dendro" or options is None:
            return False
        spec: DendroCompletionSpec | None = options.completion
        completion_file = self._registry.completion_file(task_id)
        if spec is None or completion_file is None or completion_file.is_file():
            return False
        if not spec.accept_zero_exit_code or exit_code != 0:
            return False

        if spec.success_regex is not None:
            log_path = (
                self._registry.task_directory(task_id)
                / spec.log_relative_path
            )
            if not log_path.is_file():
                return False
            text = _read_log_tail(log_path, spec.max_log_bytes)
            if re.search(spec.success_regex, text) is None:
                return False

        marker = CompletionMarker(
            task_id=task_id,
            success=True,
            completed_at_utc=datetime.now(timezone.utc),
            details={
                "source": "dendro_clean_exit",
                "exit_code": exit_code,
                "success_regex": spec.success_regex,
            },
        )
        completion_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = completion_file.with_suffix(completion_file.suffix + ".tmp")
        temporary.write_text(marker.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(completion_file)
        return True


class DendroProgressSynchronizer:
    def __init__(self, registry: PersistentTaskRegistry) -> None:
        self._registry = registry

    def refresh(self, task_id: str) -> bool:
        definition = self._registry.get_definition(task_id)
        options = definition.runtime.dendro_options
        if definition.runtime.adapter != "dendro" or options is None:
            return False
        spec: DendroProgressSpec | None = options.progress
        progress_file = self._registry.progress_file(task_id)
        if spec is None or progress_file is None:
            return False

        log_path = self._registry.task_directory(task_id) / spec.log_relative_path
        if not log_path.is_file():
            return False
        text = _read_log_tail(log_path, spec.max_log_bytes)
        # Evaluate the progress expression one log line at a time.  Real
        # Dendro checkpoint paths contain names such as ``bssn_cp_0_step.cp``.
        # A permissive expression like ``step[^0-9]*(\d+)`` can otherwise
        # cross the newline after that filename and consume the year from the
        # next timestamp (for example, 2026) as the numerical timestep.
        matches = [
            match
            for line in text.splitlines()
            for match in re.finditer(spec.step_regex, line)
        ]
        if not matches:
            return False
        match = matches[-1]
        raw = match.groupdict().get("step")
        if raw is None and match.groups():
            raw = match.group(1)
        if raw is None:
            return False
        step = float(raw)
        payload = {
            "format_version": 1,
            "task_id": task_id,
            "completed_units": step,
            "total_units": spec.total_steps,
            "updated_at_utc": datetime.fromtimestamp(
                log_path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            "details": {
                "source": "dendro_log_parser",
                "log_relative_path": spec.log_relative_path,
            },
        }
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_file.with_suffix(progress_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(progress_file)
        return True
