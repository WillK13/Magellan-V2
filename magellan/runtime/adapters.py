from __future__ import annotations

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
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> bool:
        if checkpoint_file.is_file() and checkpoint_file.stat().st_size > 0:
            return True
        if checkpoint_directory.is_dir():
            return any(
                path.is_file() and path.stat().st_size > 0
                for path in checkpoint_directory.rglob("*")
            )
        return False

    def build_launch_plan(
        self,
        spec: LocalProcessSpec,
        render: RenderValue,
        checkpoint_directory: Path,
        checkpoint_file: Path,
    ) -> RuntimeLaunchPlan:
        resumed = self._checkpoint_ready(
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
