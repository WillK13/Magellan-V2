from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


stop_requested = False


def request_stop(_signum: int, _frame: Any) -> None:
    global stop_requested
    stop_requested = True


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def atomic_torch_save(
    payload: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    torch.save(payload, temporary)
    os.replace(temporary, path)


def torch_load_compat(path: Path) -> dict:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def optimizer_to_device(
    optimizer: AdamW,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def local_model_exists(
    checkpoint_directory: Path,
) -> bool:
    config_exists = (
        checkpoint_directory / "config.json"
    ).is_file()

    weight_exists = any(
        (
            checkpoint_directory / filename
        ).is_file()
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
        )
    )

    return config_exists and weight_exists


def build_manifest(
    checkpoint_directory: Path,
    completed_steps: int,
) -> dict:
    files: list[dict] = []

    for path in sorted(checkpoint_directory.rglob("*")):
        if not path.is_file():
            continue

        if path.name == "complete.json":
            continue

        if path.name.endswith(".tmp"):
            continue

        files.append(
            {
                "path": path.relative_to(
                    checkpoint_directory
                ).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "format_version": 1,
        "workload_type": "causal-lm-training",
        "completed_steps": completed_steps,
        "files": files,
    }


def save_checkpoint(
    checkpoint_directory: Path,
    model,
    tokenizer,
    optimizer: AdamW,
    completed_steps: int,
    loss_value: float | None,
) -> None:
    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save model and tokenizer in Hugging Face local format.
    model.save_pretrained(
        checkpoint_directory,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(checkpoint_directory)

    training_state = {
        "completed_steps": completed_steps,
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }

    atomic_torch_save(
        training_state,
        checkpoint_directory / "optimizer.pt",
    )

    atomic_json_write(
        checkpoint_directory / "meta.json",
        {
            "completed_steps": completed_steps,
            "loss": loss_value,
            "node_id": os.getenv(
                "MAGELLAN_NODE_ID",
                "unknown",
            ),
            "saved_at_unix": time.time(),
        },
    )

    # The manifest is written last. Its presence means every
    # referenced file completed successfully.
    atomic_json_write(
        checkpoint_directory / "complete.json",
        build_manifest(
            checkpoint_directory,
            completed_steps,
        ),
    )

    print(
        f"[LLM] checkpoint completed_steps="
        f"{completed_steps} "
        f"directory={checkpoint_directory}",
        flush=True,
    )


def interruptible_sleep(seconds: float) -> None:
    deadline = time.monotonic() + seconds

    while (
        not stop_requested
        and time.monotonic() < deadline
    ):
        time.sleep(
            min(
                0.1,
                max(0.0, deadline - time.monotonic()),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
    )
    parser.add_argument(
        "--ready-file",
        required=True,
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000_000,
    )
    parser.add_argument(
        "--sleep-per-step",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--text",
        default=(
            "Magellan migrates stateful workloads "
            "between carbon-aware computing regions."
        ),
    )
    parser.add_argument(
        "--dataset-file",
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.completion_file is None) != (args.output_dir is None):
        raise ValueError(
            "--completion-file and --output-dir must be supplied together"
        )

    completion_file = (
        Path(args.completion_file).resolve()
        if args.completion_file is not None
        else None
    )
    output_directory = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else None
    )

    training_text = args.text

    if args.dataset_file is not None:
        dataset_path = Path(
            args.dataset_file
        ).expanduser().resolve()

        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Training dataset does not exist: "
                f"{dataset_path}"
            )

        training_text = dataset_path.read_text(
            encoding="utf-8"
        ).strip()

        if not training_text:
            raise ValueError(
                f"Training dataset is empty: "
                f"{dataset_path}"
            )

        print(
            f"[LLM] dataset={dataset_path} "
            f"bytes={dataset_path.stat().st_size}",
            flush=True,
        )

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(7)

    checkpoint_directory = Path(
        args.checkpoint_dir
    ).resolve()
    ready_file = Path(args.ready_file).resolve()

    checkpoint_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    ready_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    resumed = local_model_exists(
        checkpoint_directory
    )

    if resumed:
        print(
            f"[LLM] loading local checkpoint "
            f"from {checkpoint_directory}",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_directory
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_directory
        )
    else:
        print(
            f"[LLM] loading base model {args.model}",
            flush=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model
        )

    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    completed_steps = 0
    optimizer_path = (
        checkpoint_directory / "optimizer.pt"
    )

    if resumed and optimizer_path.is_file():
        saved_state = torch_load_compat(
            optimizer_path
        )

        completed_steps = int(
            saved_state.get("completed_steps", 0)
        )

        optimizer.load_state_dict(
            saved_state["optimizer_state_dict"]
        )
        optimizer_to_device(optimizer, device)

        rng_state = saved_state.get(
            "torch_rng_state"
        )

        if rng_state is not None:
            torch.set_rng_state(rng_state)

        print(
            f"[LLM] resumed completed_steps="
            f"{completed_steps}",
            flush=True,
        )

    inputs = tokenizer(
        training_text,
        return_tensors="pt",
    ).to(device)

    labels = inputs["input_ids"].clone()

    # A readiness marker always implies a complete, validated
    # checkpoint exists—even on the first startup.
    save_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        completed_steps=completed_steps,
        loss_value=None,
    )

    atomic_json_write(
        ready_file,
        {
            "ready": True,
            "node_id": os.getenv(
                "MAGELLAN_NODE_ID",
                "unknown",
            ),
            "pid": os.getpid(),
            "device": str(device),
            "resumed": resumed,
            "completed_steps": completed_steps,
            "ready_at_unix": time.time(),
        },
    )

    print(
        f"[LLM] ready node="
        f"{os.getenv('MAGELLAN_NODE_ID')} "
        f"completed_steps={completed_steps}",
        flush=True,
    )

    last_loss: float | None = None

    while (
        completed_steps < args.max_steps
        and not stop_requested
    ):
        model.train()

        outputs = model(
            **inputs,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_steps += 1
        last_loss = float(loss.item())

        print(
            f"[LLM] step={completed_steps} "
            f"loss={last_loss:.6f} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )

        save_checkpoint(
            checkpoint_directory=checkpoint_directory,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            completed_steps=completed_steps,
            loss_value=last_loss,
        )

        interruptible_sleep(
            args.sleep_per_step
        )

    # SIGTERM requests a graceful stop. Write one final complete
    # checkpoint before the process exits.
    save_checkpoint(
        checkpoint_directory=checkpoint_directory,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        completed_steps=completed_steps,
        loss_value=last_loss,
    )

    natural_completion = (
        not stop_requested
        and completed_steps >= args.max_steps
    )

    if natural_completion:
        assert completion_file is not None
        assert output_directory is not None
        output_directory.mkdir(parents=True, exist_ok=True)
        completed_at = datetime.now(timezone.utc)

        atomic_json_write(
            output_directory / "training-summary.json",
            {
                "task_id": os.getenv(
                    "MAGELLAN_TASK_ID",
                    "unknown",
                ),
                "completed_steps": completed_steps,
                "loss": last_loss,
                "model": args.model,
                "device": str(device),
                "node_id": os.getenv(
                    "MAGELLAN_NODE_ID",
                    "unknown",
                ),
                "completed_at_utc": completed_at.isoformat(),
            },
        )

        # Written last: this marker commits successful completion.
        atomic_json_write(
            completion_file,
            {
                "format_version": 1,
                "task_id": os.getenv(
                    "MAGELLAN_TASK_ID",
                    "unknown",
                ),
                "success": True,
                "completed_at_utc": completed_at.isoformat(),
                "details": {
                    "completed_steps": completed_steps,
                    "loss": last_loss,
                    "node_id": os.getenv(
                        "MAGELLAN_NODE_ID",
                        "unknown",
                    ),
                },
            },
        )

        print(
            f"[LLM] completed completed_steps="
            f"{completed_steps} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )
    else:
        print(
            f"[LLM] stopped completed_steps="
            f"{completed_steps} "
            f"node={os.getenv('MAGELLAN_NODE_ID')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
