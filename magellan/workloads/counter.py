from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


stop_requested = False


def request_stop(_signum, _frame) -> None:
    global stop_requested
    stop_requested = True


def atomic_write(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            {
                "value": value,
                "updated_at_unix": time.time(),
                "node_id": os.getenv(
                    "MAGELLAN_NODE_ID",
                    "unknown",
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def load_value(path: Path) -> int:
    if not path.exists():
        return 0

    raw = json.loads(path.read_text(encoding="utf-8"))
    return int(raw.get("value", 0))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-file",
        required=True,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_file)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    value = load_value(checkpoint_path)

    print(
        f"[counter] resumed value={value} "
        f"node={os.getenv('MAGELLAN_NODE_ID')}",
        flush=True,
    )

    while not stop_requested:
        value += 1
        atomic_write(checkpoint_path, value)

        print(
            f"[counter] value={value} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )

        time.sleep(args.interval_seconds)

    atomic_write(checkpoint_path, value)

    print(
        f"[counter] stopped value={value}",
        flush=True,
    )


if __name__ == "__main__":
    main()
