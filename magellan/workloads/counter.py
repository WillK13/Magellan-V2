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


def write_progress(
    path: Path | None,
    value: int,
    max_value: int | None,
) -> None:
    if path is None:
        return

    atomic_json_write(
        path,
        {
            "format_version": 1,
            "task_id": os.getenv(
                "MAGELLAN_TASK_ID",
                "unknown",
            ),
            "completed_units": value,
            "total_units": max_value,
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "node_id": os.getenv(
                "MAGELLAN_NODE_ID",
                "unknown",
            ),
            "details": {
                "unit": "counter-value",
            },
        },
    )


def ensure_checkpoint_padding(
    checkpoint_path: Path,
    padding_bytes: int,
    chunk_bytes: int = 1024 * 1024,
) -> Path | None:
    if padding_bytes <= 0:
        return None

    padding_path = checkpoint_path.parent / "payload.bin"
    if padding_path.is_file() and padding_path.stat().st_size == padding_bytes:
        return padding_path

    padding_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = padding_path.with_suffix(".bin.tmp")
    remaining = padding_bytes
    with temporary.open("wb") as handle:
        while remaining > 0:
            size = min(chunk_bytes, remaining)
            handle.write(os.urandom(size))
            remaining -= size
    os.replace(temporary, padding_path)
    return padding_path


def completion_target(
    *,
    current_value: int,
    configured_max_value: int | None,
    current_node_id: str,
    initial_node_id: str | None,
    complete_after_migration_steps: int | None,
) -> int | None:
    if initial_node_id is None or complete_after_migration_steps is None:
        return configured_max_value
    if current_node_id == initial_node_id:
        return configured_max_value
    if complete_after_migration_steps < 1:
        raise ValueError("--complete-after-migration-steps must be positive")
    return current_value + complete_after_migration_steps


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
        "--checkpoint-padding-bytes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=None,
    )
    parser.add_argument("--initial-node-id", default=None)
    parser.add_argument(
        "--complete-after-migration-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--progress-file",
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
    progress_path = (
        Path(args.progress_file)
        if args.progress_file is not None
        else None
    )
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

    if args.checkpoint_padding_bytes < 0:
        raise ValueError("--checkpoint-padding-bytes must be non-negative")
    padding_path = ensure_checkpoint_padding(
        checkpoint_path,
        args.checkpoint_padding_bytes,
    )
    value = load_value(checkpoint_path)
    current_node_id = os.getenv("MAGELLAN_NODE_ID", "unknown")
    effective_max_value = completion_target(
        current_value=value,
        configured_max_value=args.max_value,
        current_node_id=current_node_id,
        initial_node_id=args.initial_node_id,
        complete_after_migration_steps=args.complete_after_migration_steps,
    )

    if padding_path is not None:
        print(
            f"[counter] checkpoint-padding bytes={padding_path.stat().st_size}",
            flush=True,
        )

    print(
        f"[counter] resumed value={value} "
        f"node={os.getenv('MAGELLAN_NODE_ID')}",
        flush=True,
    )

    write_progress(
        progress_path,
        value,
        effective_max_value,
    )

    while not stop_requested:
        if effective_max_value is not None and value >= effective_max_value:
            break

        value += 1
        atomic_write(checkpoint_path, value)
        write_progress(
            progress_path,
            value,
            effective_max_value,
        )

        print(
            f"[counter] value={value} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )

        time.sleep(args.interval_seconds)

    atomic_write(checkpoint_path, value)
    write_progress(
        progress_path,
        value,
        effective_max_value,
    )

    natural_completion = (
        not stop_requested
        and effective_max_value is not None
        and value >= effective_max_value
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
