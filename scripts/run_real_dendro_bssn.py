#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


def _replace_toml_value(text: str, key: str, value: str) -> str:
    pattern = rf"(?m)^\s*{re.escape(key)}\s*=.*$"
    updated, count = re.subn(pattern, f"{key} = {value}", text)
    if count != 1:
        raise RuntimeError(f"{key}: expected one TOML assignment, found {count}")
    return updated


def render_runtime_parameters(
    *,
    template_path: Path,
    output_path: Path,
    checkpoint_directory: Path,
    output_directory: Path,
    resume: bool,
) -> Path:
    text = template_path.read_text(encoding="utf-8")
    checkpoint_prefix = checkpoint_directory / "bssn_cp"
    vtu_prefix = output_directory / "vtu" / "bssn_gr"
    profile_prefix = output_directory / "dat" / "dgr"
    for directory in (
        checkpoint_directory,
        vtu_prefix.parent,
        profile_prefix.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    replacements = {
        "BSSN_RESTORE_SOLVER": "1" if resume else "0",
        "BSSN_CHKPT_FILE_PREFIX": repr(str(checkpoint_prefix)),
        "BSSN_VTU_FILE_PREFIX": repr(str(vtu_prefix)),
        "BSSN_PROFILE_FILE_PREFIX": repr(str(profile_prefix)),
    }
    for key, value in replacements.items():
        text = _replace_toml_value(text, key, value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch upstream Dendro-GR BSSN with Magellan paths"
    )
    parser.add_argument("solver")
    parser.add_argument("parameter_template")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--ts-mode", default="1")
    args = parser.parse_args()

    checkpoint_directory = Path(
        os.environ["MAGELLAN_DENDRO_CHECKPOINT_DIRECTORY"]
    ).resolve()
    task_directory = checkpoint_directory.parent
    output_directory = task_directory / "output"
    parameter_path = task_directory / "dendro-runtime.toml"
    resume = os.environ.get("MAGELLAN_DENDRO_RESUME") == "1"
    render_runtime_parameters(
        template_path=Path(args.parameter_template).resolve(),
        output_path=parameter_path,
        checkpoint_directory=checkpoint_directory,
        output_directory=output_directory,
        resume=resume,
    )
    command = [
        "mpirun",
        "--use-hwthread-cpus",
        "-np",
        str(args.world_size),
        str(Path(args.solver).resolve()),
        str(parameter_path),
        args.ts_mode,
    ]
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
