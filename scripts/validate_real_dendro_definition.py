from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from magellan.runtime.dendro import DendroCheckpointDiscovery  # noqa: E402
from magellan.submission.models import TaskDefinitionSubmission  # noqa: E402


UNRESOLVED_MARKERS = ("<SET_", "REPLACE_WITH", "TODO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a real Dendro-GR Magellan task definition"
    )
    parser.add_argument("--definition", required=True)
    parser.add_argument("--checkpoint-directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.definition)
    definition = TaskDefinitionSubmission.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    runtime = definition.runtime
    if runtime.adapter != "dendro":
        raise SystemExit("Definition does not use the dendro adapter")
    options = runtime.dendro_options
    if options is None or options.checkpoint_discovery is None:
        raise SystemExit("Definition has no Dendro checkpoint discovery policy")

    rendered = json.dumps(definition.model_dump(mode="json"))
    unresolved = [item for item in UNRESOLVED_MARKERS if item in rendered]
    if unresolved:
        raise SystemExit(
            "Definition still contains unresolved template markers: "
            + ", ".join(unresolved)
        )

    command = runtime.command[0]
    executable = Path(command)
    if executable.is_absolute():
        if not executable.is_file():
            raise SystemExit(f"Executable does not exist: {executable}")
    elif shutil.which(command) is None:
        raise SystemExit(f"Command is not available on PATH: {command}")

    if args.checkpoint_directory:
        directory = Path(args.checkpoint_directory).resolve()
        manifest = directory / runtime.checkpoint_manifest_relative_path
        discovered = DendroCheckpointDiscovery().discover(
            checkpoint_directory=directory,
            manifest_path=manifest,
            spec=options.checkpoint_discovery,
        )
        if discovered is None:
            raise SystemExit("No complete stable Dendro checkpoint was found")
        print(f"checkpoint_manifest={discovered}")

    print(f"definition_id={definition.definition_id}")
    print(f"command={' '.join(runtime.command)}")
    print(f"minimum_process_count={runtime.minimum_process_count}")
    print("REAL_DENDRO_DEFINITION_VALID")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
