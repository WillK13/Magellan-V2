from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from magellan.state.task_models import LocalProcessSpec


RenderValue = Callable[[str], str]


@dataclass(frozen=True)
class RuntimeLaunchPlan:
    adapter: str
    command: list[str]
    resumed_from_checkpoint: bool = False
    environment: dict[str, str] | None = None


class RuntimeAdapter(Protocol):
    name: str

    def build_launch_plan(
        self,
        spec: LocalProcessSpec,
        render: RenderValue,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> RuntimeLaunchPlan:
        ...


class PythonModuleAdapter:
    name = "python_module"

    def build_launch_plan(
        self,
        spec: LocalProcessSpec,
        render: RenderValue,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> RuntimeLaunchPlan:
        del checkpoint_directory, checkpoint_file
        assert spec.module is not None
        return RuntimeLaunchPlan(
            adapter=self.name,
            command=[
                sys.executable,
                "-m",
                spec.module,
                *[render(item) for item in spec.arguments],
            ],
        )


class CommandAdapter:
    name = "command"

    def build_launch_plan(
        self,
        spec: LocalProcessSpec,
        render: RenderValue,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> RuntimeLaunchPlan:
        del checkpoint_directory, checkpoint_file
        return RuntimeLaunchPlan(
            adapter=self.name,
            command=[
                *[render(item) for item in spec.command],
                *[render(item) for item in spec.arguments],
            ],
        )


class DendroCommandAdapter:
    """Application-checkpoint adapter for Dendro-GR style executables.

    The executable owns checkpoint creation. Magellan validates and transfers
    the manifest-based checkpoint, then appends ``resume_arguments`` when a
    transferred checkpoint is present on the destination.
    """

    name = "dendro"

    @staticmethod
    def _checkpoint_ready(
        spec: LocalProcessSpec,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> tuple[bool, int | None]:
        manifest_relative = spec.checkpoint_manifest_relative_path
        if manifest_relative is not None:
            manifest_path = checkpoint_directory / manifest_relative
            if not manifest_path.is_file():
                return False, None
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                return False, None
            files = manifest.get("files")
            if not isinstance(files, list) or not files:
                return False, None
            for item in files:
                if not isinstance(item, dict):
                    return False, None
                relative = item.get("path")
                expected_size = item.get("size_bytes")
                if not isinstance(relative, str) or not isinstance(
                    expected_size, int
                ):
                    return False, None
                path = checkpoint_directory / relative
                if not path.is_file() or path.stat().st_size != expected_size:
                    return False, None
            step = manifest.get("checkpoint_step")
            return True, step if isinstance(step, int) else None

        if checkpoint_file.is_file() and checkpoint_file.stat().st_size > 0:
            return True, None
        if checkpoint_directory.is_dir():
            ready = any(
                path.is_file() and path.stat().st_size > 0
                for path in checkpoint_directory.rglob("*")
            )
            return ready, None
        return False, None

    def build_launch_plan(
        self,
        spec: LocalProcessSpec,
        render: RenderValue,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> RuntimeLaunchPlan:
        resumed, checkpoint_step = self._checkpoint_ready(
            spec,
            checkpoint_directory,
            checkpoint_file,
        )
        arguments = list(spec.arguments)
        if resumed:
            arguments.extend(spec.resume_arguments)
        return RuntimeLaunchPlan(
            adapter=self.name,
            command=[
                *[render(item) for item in spec.command],
                *[render(item) for item in arguments],
            ],
            resumed_from_checkpoint=resumed,
            environment={
                "MAGELLAN_DENDRO_RESUME": "1" if resumed else "0",
                "MAGELLAN_DENDRO_CHECKPOINT_DIRECTORY": str(
                    checkpoint_directory
                ),
                "MAGELLAN_DENDRO_CHECKPOINT_STEP": (
                    str(checkpoint_step)
                    if checkpoint_step is not None
                    else ""
                ),
            },
        )


class RuntimeAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, RuntimeAdapter] = {
            adapter.name: adapter
            for adapter in (
                PythonModuleAdapter(),
                CommandAdapter(),
                DendroCommandAdapter(),
            )
        }

    def get(self, name: str) -> RuntimeAdapter:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise RuntimeError(f"Unknown runtime adapter: {name}") from exc

    def names(self) -> list[str]:
        return sorted(self._adapters)
