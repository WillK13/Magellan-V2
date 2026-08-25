from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


stop_requested = False


def request_stop(_signum, _frame) -> None:
    global stop_requested
    stop_requested = True


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def problem_scale(size: str) -> int:
    try:
        return {"small": 1, "medium": 2, "large": 3}[size]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark size: {size}") from exc


def json_kernel(size: str, seed: int, iteration: int) -> float:
    scale = problem_scale(size)
    count = 64 * scale
    payload = [
        {
            "id": index,
            "seed": seed,
            "iteration": iteration,
            "name": f"record-{index:04d}",
            "values": [
                (seed * 31 + iteration * 17 + index * 13 + offset) % 10007
                for offset in range(8 * scale)
            ],
            "active": index % 3 != 0,
        }
        for index in range(count)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    decoded = json.loads(encoded)
    return float(
        len(encoded)
        + sum(item["values"][0] for item in decoded)
    )


def matmul_kernel(size: str, seed: int, iteration: int) -> float:
    scale = problem_scale(size)
    dimension = 14 + 6 * scale
    base = seed + iteration
    a = [
        [((row * 17 + col * 11 + base) % 101) / 101.0 for col in range(dimension)]
        for row in range(dimension)
    ]
    b = [
        [((row * 7 + col * 19 + base * 3) % 103) / 103.0 for col in range(dimension)]
        for row in range(dimension)
    ]
    checksum = 0.0
    for row in range(dimension):
        for col in range(dimension):
            value = 0.0
            for inner in range(dimension):
                value += a[row][inner] * b[inner][col]
            checksum += value
    return checksum


# Five-body solar-system kernel following the standard benchmark-game/
# pyperformance n-body shape, adapted so one outer Magellan iteration is a
# durable progress unit.
_NBODY_INITIAL = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.47841760435743),
    (
        4.841431442464721,
        -1.1603200440274284,
        -0.10362204447112311,
        0.001660076642744037 * 365.24,
        0.007699011184197404 * 365.24,
        -0.0000690460016972063 * 365.24,
        0.0009547919384243266 * 39.47841760435743,
    ),
    (
        8.34336671824458,
        4.124798564124305,
        -0.4035234171143214,
        -0.002767425107268624 * 365.24,
        0.004998528012349172 * 365.24,
        0.00002304172975737639 * 365.24,
        0.0002858859806661308 * 39.47841760435743,
    ),
    (
        12.894369562139131,
        -15.111151401698631,
        -0.22330757889265573,
        0.002964601375647616 * 365.24,
        0.0023784717395948095 * 365.24,
        -0.000029658956854023756 * 365.24,
        0.00004366244043351563 * 39.47841760435743,
    ),
    (
        15.379697114850917,
        -25.919314609987964,
        0.17925877295037118,
        0.0026806777249038932 * 365.24,
        0.001628241700382423 * 365.24,
        -0.00009515922545197159 * 365.24,
        0.000051513890204661145 * 39.47841760435743,
    ),
)


def _nbody_energy(bodies: list[list[float]]) -> float:
    energy = 0.0
    for index, body in enumerate(bodies):
        x, y, z, vx, vy, vz, mass = body
        energy += 0.5 * mass * (vx * vx + vy * vy + vz * vz)
        for other in bodies[index + 1 :]:
            dx = x - other[0]
            dy = y - other[1]
            dz = z - other[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            energy -= mass * other[6] / distance
    return energy


def nbody_kernel(size: str, seed: int, iteration: int) -> float:
    scale = problem_scale(size)
    bodies = [list(item) for item in _NBODY_INITIAL]
    # Keep the workload deterministic while avoiding identical floating-point
    # paths for every outer iteration.
    dt = 0.01 + ((seed + iteration) % 5) * 0.0001
    steps = 100 * scale
    for _ in range(steps):
        for index, body in enumerate(bodies):
            for other in bodies[index + 1 :]:
                dx = body[0] - other[0]
                dy = body[1] - other[1]
                dz = body[2] - other[2]
                distance_sq = dx * dx + dy * dy + dz * dz
                magnitude = dt / (distance_sq * math.sqrt(distance_sq))
                body[3] -= dx * other[6] * magnitude
                body[4] -= dy * other[6] * magnitude
                body[5] -= dz * other[6] * magnitude
                other[3] += dx * body[6] * magnitude
                other[4] += dy * body[6] * magnitude
                other[5] += dz * body[6] * magnitude
        for body in bodies:
            body[0] += dt * body[3]
            body[1] += dt * body[4]
            body[2] += dt * body[5]
    return _nbody_energy(bodies)


KERNELS: dict[str, Callable[[str, int, int], float]] = {
    "nbody": nbody_kernel,
    "json": json_kernel,
    "matmul": matmul_kernel,
}


def load_state(path: Path) -> tuple[int, float]:
    if not path.is_file():
        return 0, 0.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("completed_iterations", 0)), float(
        payload.get("checksum", 0.0)
    )


def state_payload(
    *,
    benchmark: str,
    size: str,
    seed: int,
    completed: int,
    total: int,
    checksum: float,
) -> dict:
    return {
        "format_version": 1,
        "benchmark": benchmark,
        "size": size,
        "seed": seed,
        "completed_iterations": completed,
        "total_iterations": total,
        "checksum": checksum,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "node_id": os.getenv("MAGELLAN_NODE_ID", "unknown"),
    }


def write_progress(
    path: Path | None,
    *,
    completed: int,
    total: int,
    benchmark: str,
    size: str,
) -> None:
    if path is None:
        return
    atomic_json_write(
        path,
        {
            "format_version": 1,
            "task_id": os.getenv("MAGELLAN_TASK_ID", "unknown"),
            "completed_units": completed,
            "total_units": total,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "node_id": os.getenv("MAGELLAN_NODE_ID", "unknown"),
            "details": {
                "unit": "benchmark-iteration",
                "benchmark": benchmark,
                "size": size,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpointable standard CPU benchmark workload for Magellan "
            "population and contention experiments"
        )
    )
    parser.add_argument("--benchmark", choices=sorted(KERNELS), required=True)
    parser.add_argument("--size", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--completion-file", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if (args.completion_file is None) != (args.output_dir is None):
        raise ValueError(
            "--completion-file and --output-dir must be supplied together"
        )

    checkpoint = Path(args.checkpoint_file)
    progress = Path(args.progress_file) if args.progress_file else None
    completion = Path(args.completion_file) if args.completion_file else None
    output = Path(args.output_dir) if args.output_dir else None

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    completed, checksum = load_state(checkpoint)
    if completed > args.iterations:
        raise ValueError(
            "Checkpoint completed_iterations exceeds configured --iterations"
        )

    print(
        f"[benchmark] type={args.benchmark} size={args.size} "
        f"resumed={completed} total={args.iterations} "
        f"node={os.getenv('MAGELLAN_NODE_ID', 'unknown')}",
        flush=True,
    )
    write_progress(
        progress,
        completed=completed,
        total=args.iterations,
        benchmark=args.benchmark,
        size=args.size,
    )

    kernel = KERNELS[args.benchmark]
    started = time.monotonic()
    while completed < args.iterations and not stop_requested:
        checksum += kernel(args.size, args.seed, completed)
        completed += 1
        atomic_json_write(
            checkpoint,
            state_payload(
                benchmark=args.benchmark,
                size=args.size,
                seed=args.seed,
                completed=completed,
                total=args.iterations,
                checksum=checksum,
            ),
        )
        write_progress(
            progress,
            completed=completed,
            total=args.iterations,
            benchmark=args.benchmark,
            size=args.size,
        )
        if completed == 1 or completed % 25 == 0:
            print(
                f"[benchmark] completed={completed}/{args.iterations} "
                f"type={args.benchmark} size={args.size}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    atomic_json_write(
        checkpoint,
        state_payload(
            benchmark=args.benchmark,
            size=args.size,
            seed=args.seed,
            completed=completed,
            total=args.iterations,
            checksum=checksum,
        ),
    )
    write_progress(
        progress,
        completed=completed,
        total=args.iterations,
        benchmark=args.benchmark,
        size=args.size,
    )

    if not stop_requested and completed >= args.iterations:
        if completion is None or output is None:
            return
        output.mkdir(parents=True, exist_ok=True)
        completed_at = datetime.now(timezone.utc)
        atomic_json_write(
            output / "result.json",
            {
                "task_id": os.getenv("MAGELLAN_TASK_ID", "unknown"),
                "benchmark": args.benchmark,
                "size": args.size,
                "seed": args.seed,
                "iterations": completed,
                "checksum": checksum,
                "active_runtime_seconds": elapsed,
                "node_id": os.getenv("MAGELLAN_NODE_ID", "unknown"),
                "completed_at_utc": completed_at.isoformat(),
            },
        )
        atomic_json_write(
            completion,
            {
                "format_version": 1,
                "task_id": os.getenv("MAGELLAN_TASK_ID", "unknown"),
                "success": True,
                "completed_at_utc": completed_at.isoformat(),
                "details": {
                    "benchmark": args.benchmark,
                    "size": args.size,
                    "seed": args.seed,
                    "iterations": completed,
                },
            },
        )


if __name__ == "__main__":
    main()
