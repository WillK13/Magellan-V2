from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def rank_worker(rank: int, stop_event: mp.Event) -> None:
    value = rank + 1
    while not stop_event.is_set():
        # Keep workers alive and measurable without consuming an entire core.
        value = (value * 1103515245 + 12345) % (2**31)
        time.sleep(0.05)


def write_checkpoint(
    checkpoint_directory: Path,
    checkpoint_file: Path,
    manifest_file: Path,
    step: int,
    world_size: int,
    task_id: str,
    node_id: str,
) -> None:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "task_id": task_id,
        "step": step,
        "world_size": world_size,
        "node_id": node_id,
        "written_at_utc": utc_now(),
    }
    atomic_json(checkpoint_file, state)

    files = [checkpoint_file]
    for rank in range(world_size):
        rank_file = checkpoint_directory / f"rank-{rank:04d}.bin"
        temporary = rank_file.with_suffix(".bin.tmp")
        temporary.write_bytes(
            f"task={task_id}\nstep={step}\nrank={rank}\n".encode()
        )
        os.replace(temporary, rank_file)
        files.append(rank_file)

    manifest = {
        "format_version": 1,
        "task_id": task_id,
        "step": step,
        "world_size": world_size,
        "files": [
            {
                "path": path.relative_to(checkpoint_directory).as_posix(),
                "size_bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    atomic_json(manifest_file, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--completion-file", required=True)
    parser.add_argument("--readiness-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--max-step", type=int, default=240)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task_id = os.environ.get("MAGELLAN_TASK_ID", "dendro-task")
    node_id = os.environ.get("MAGELLAN_NODE_ID", "unknown")
    checkpoint_file = Path(args.checkpoint_file)
    checkpoint_directory = checkpoint_file.parent
    manifest_file = Path(args.checkpoint_manifest)
    progress_file = Path(args.progress_file)
    completion_file = Path(args.completion_file)
    readiness_file = Path(args.readiness_file)
    output_directory = Path(args.output_dir)

    step = 0
    if checkpoint_file.is_file():
        try:
            previous = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            step = int(previous.get("step", 0))
        except (ValueError, json.JSONDecodeError):
            step = 0

    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    worker_stop = mp.Event()
    workers = [
        mp.Process(target=rank_worker, args=(rank, worker_stop))
        for rank in range(args.world_size)
    ]
    for worker in workers:
        worker.start()

    readiness_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        readiness_file,
        {
            "task_id": task_id,
            "node_id": node_id,
            "world_size": args.world_size,
            "pid": os.getpid(),
            "resumed": bool(args.resume or step > 0),
            "ready_at_utc": utc_now(),
        },
    )

    try:
        while step < args.max_step and not stop_requested:
            step += 1
            if step % args.checkpoint_every == 0:
                write_checkpoint(
                    checkpoint_directory,
                    checkpoint_file,
                    manifest_file,
                    step,
                    args.world_size,
                    task_id,
                    node_id,
                )
            atomic_json(
                progress_file,
                {
                    "format_version": 1,
                    "task_id": task_id,
                    "completed_units": step,
                    "total_units": args.max_step,
                    "updated_at_utc": utc_now(),
                    "node_id": node_id,
                    "details": {
                        "runtime": "dendro-compatible-mpi-harness",
                        "world_size": args.world_size,
                        "resumed": bool(args.resume),
                    },
                },
            )
            time.sleep(args.interval_seconds)

        write_checkpoint(
            checkpoint_directory,
            checkpoint_file,
            manifest_file,
            step,
            args.world_size,
            task_id,
            node_id,
        )

        if stop_requested:
            return 0

        output_directory.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output_directory / "dendro-result.json",
            {
                "task_id": task_id,
                "final_step": step,
                "world_size": args.world_size,
                "node_id": node_id,
            },
        )
        atomic_json(
            completion_file,
            {
                "format_version": 1,
                "task_id": task_id,
                "success": True,
                "completed_at_utc": utc_now(),
                "details": {
                    "final_step": step,
                    "world_size": args.world_size,
                    "runtime": "dendro-compatible-mpi-harness",
                },
            },
        )
        return 0
    finally:
        worker_stop.set()
        for worker in workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
