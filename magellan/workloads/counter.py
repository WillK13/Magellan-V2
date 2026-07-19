from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


stop_requested = False


def request_stop(_signum, _frame) -> None:
    global stop_requested
    stop_requested = True


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write(path: Path, value: int) -> None:
    atomic_json_write(
        path,
        {
            "value": value,
            "updated_at_unix": time.time(),
            "node_id": os.getenv(
                "MAGELLAN_NODE_ID",
                "unknown",
            ),
        },
    )


def load_value(path: Path) -> int:
    if not path.exists():
        return 0

    raw = json.loads(path.read_text(encoding="utf-8"))
    return int(raw.get("value", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--completion-file",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    args = parser.parse_args()

    if (args.completion_file is None) != (args.output_dir is None):
        raise ValueError(
            "--completion-file and --output-dir must be supplied together"
        )

    checkpoint_path = Path(args.checkpoint_file)
    completion_path = (
        Path(args.completion_file)
        if args.completion_file is not None
        else None
    )
    output_directory = (
        Path(args.output_dir)
        if args.output_dir is not None
        else None
    )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    value = load_value(checkpoint_path)

    print(
        f"[counter] resumed value={value} "
        f"node={os.getenv('MAGELLAN_NODE_ID')}",
        flush=True,
    )

    while not stop_requested:
        if args.max_value is not None and value >= args.max_value:
            break

        value += 1
        atomic_write(checkpoint_path, value)

        print(
            f"[counter] value={value} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )

        time.sleep(args.interval_seconds)

    atomic_write(checkpoint_path, value)

    natural_completion = (
        not stop_requested
        and args.max_value is not None
        and value >= args.max_value
    )

    if natural_completion:
        assert completion_path is not None
        assert output_directory is not None
        output_directory.mkdir(parents=True, exist_ok=True)
        completed_at = datetime.now(timezone.utc)

        atomic_json_write(
            output_directory / "result.json",
            {
                "task_id": os.getenv(
                    "MAGELLAN_TASK_ID",
                    "unknown",
                ),
                "final_value": value,
                "node_id": os.getenv(
                    "MAGELLAN_NODE_ID",
                    "unknown",
                ),
                "completed_at_utc": completed_at.isoformat(),
            },
        )

        # Written last: this is the success commit record.
        atomic_json_write(
            completion_path,
            {
                "format_version": 1,
                "task_id": os.getenv(
                    "MAGELLAN_TASK_ID",
                    "unknown",
                ),
                "success": True,
                "completed_at_utc": completed_at.isoformat(),
                "details": {
                    "final_value": value,
                    "node_id": os.getenv(
                        "MAGELLAN_NODE_ID",
                        "unknown",
                    ),
                },
            },
        )

        print(
            f"[counter] completed value={value}",
            flush=True,
        )
    else:
        print(
            f"[counter] stopped value={value}",
            flush=True,
        )


if __name__ == "__main__":
    main()
