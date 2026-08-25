#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from magellan.config.loader import load_cluster_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-stage the exact CPU LLM runtime and a local Hugging Face model "
            "snapshot on every Stage-4 GCP worker, then run a real forward/backward smoke test."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--source-node-id", default="boston")
    parser.add_argument("--ssh-user", default="WILL")
    parser.add_argument("--remote-repo", default="~/Magellan-V2")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument(
        "--asset-root",
        default="experiment-assets/models",
        help="Repository-relative directory used for immutable local model snapshots.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser.parse_args()


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError("Model name does not produce a safe asset directory")
    return cleaned


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def remote_cd(path: str) -> str:
    if path == "~":
        return 'cd "$HOME"'
    if path.startswith("~/"):
        return f'cd "$HOME"/{shlex.quote(path[2:])}'
    return f"cd {shlex.quote(path)}"


def run_remote(*, node: Any, source_node_id: str, ssh_user: str, command: str, timeout: float) -> subprocess.CompletedProcess[str]:
    if node.id == source_node_id:
        argv = ["bash", "-lc", command]
    else:
        argv = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8",
            f"{ssh_user}@{node.internal_ip}",
            command,
        ]
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout)


def dependency_versions(repo: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json, torch, transformers; "
        "print(json.dumps({'torch': torch.__version__, 'transformers': transformers.__version__, "
        "'safetensors': m.version('safetensors')}))"
    )
    result = subprocess.run(
        [str(repo / ".venv/bin/python"), "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def ensure_source_snapshot(repo: Path, model: str, destination: Path) -> None:
    if (destination / "config.json").is_file() and any(
        (destination / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return
    destination.mkdir(parents=True, exist_ok=True)
    code = r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
model_name, destination = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.save_pretrained(destination)
model.save_pretrained(destination, safe_serialization=True)
print("MODEL_SNAPSHOT_READY")
'''
    subprocess.run(
        [str(repo / ".venv/bin/python"), "-c", code, model, str(destination)],
        check=True,
    )


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)
    source = cluster.get_node(args.source_node_id)
    if source.id != args.source_node_id:
        raise RuntimeError("Invalid source node")

    repo = Path.cwd().resolve()
    if not (repo / ".venv/bin/python").is_file():
        raise RuntimeError("Run from the Magellan repository with .venv installed")
    asset_relative = Path(args.asset_root) / slug(args.model)
    asset = repo / asset_relative

    dependencies = dependency_versions(repo)
    ensure_source_snapshot(repo, args.model, asset)
    source_hash = directory_sha256(asset)
    torch_base = dependencies["torch"].split("+", 1)[0]

    print(f"source_node={source.id}")
    print(f"model={args.model}")
    print(f"asset={asset}")
    print(f"asset_sha256={source_hash}")
    print(f"dependencies={json.dumps(dependencies, sort_keys=True)}")

    smoke_code = r'''
import hashlib, json, pathlib, sys
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
root = pathlib.Path(sys.argv[1])
d = hashlib.sha256()
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    rel = path.relative_to(root).as_posix().encode()
    d.update(len(rel).to_bytes(8, "big")); d.update(rel)
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(1024 * 1024), b""):
            d.update(chunk)
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True)
encoded = tok("Magellan Stage 4 calibration", return_tensors="pt")
labels = encoded["input_ids"].clone()
loss = model(**encoded, labels=labels).loss
opt = AdamW(model.parameters(), lr=5e-5)
loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
print(json.dumps({"asset_sha256": d.hexdigest(), "loss": float(loss.detach()), "torch": torch.__version__}))
'''

    for node in cluster.nodes:
        print(f"\n== llm provision {node.id} ==", flush=True)
        if node.id != source.id:
            install = f"""
set -euo pipefail
{remote_cd(args.remote_repo)}
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu {shlex.quote('torch==' + torch_base)}
python -m pip install {shlex.quote('transformers==' + dependencies['transformers'])} {shlex.quote('safetensors==' + dependencies['safetensors'])}
mkdir -p {shlex.quote(str(asset_relative.parent))}
""".strip()
            result = run_remote(
                node=node,
                source_node_id=source.id,
                ssh_user=args.ssh_user,
                command=install,
                timeout=args.timeout_seconds,
            )
            if result.stdout:
                print(result.stdout.strip())
            subprocess.run(
                [
                    "rsync", "-a", "--delete",
                    f"{asset}/",
                    f"{args.ssh_user}@{node.internal_ip}:{args.remote_repo}/{asset_relative}/",
                ],
                check=True,
                timeout=args.timeout_seconds,
            )

        probe = f"""
set -euo pipefail
{remote_cd(args.remote_repo)}
source .venv/bin/activate
python -c {shlex.quote(smoke_code)} {shlex.quote(str(asset_relative))}
""".strip()
        result = run_remote(
            node=node,
            source_node_id=source.id,
            ssh_user=args.ssh_user,
            command=probe,
            timeout=args.timeout_seconds,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if payload["asset_sha256"] != source_hash:
            raise RuntimeError(
                f"{node.id} model hash mismatch: {payload['asset_sha256']} != {source_hash}"
            )
        if not str(payload["torch"]).startswith(torch_base):
            raise RuntimeError(f"{node.id} torch mismatch: {payload['torch']} != {torch_base}")
        print(
            f"[PASS] {node.id:16} model_sha256={source_hash[:12]} "
            f"torch={payload['torch']} loss={payload['loss']:.6f}"
        )

    print("\nSEVEN_NODE_LLM_PROVISION_PASS")
    print(f"model_path={asset_relative}")
    print(f"model_sha256={source_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
