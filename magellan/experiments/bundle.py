from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, values: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def build_checksums(bundle_dir: str | Path) -> dict[str, str]:
    root = Path(bundle_dir)
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "checksums.sha256":
            continue
        result[relative] = sha256_file(path)
    return result


def write_checksums(bundle_dir: str | Path) -> dict[str, str]:
    root = Path(bundle_dir)
    checksums = build_checksums(root)
    lines = [f"{digest}  {relative}" for relative, digest in checksums.items()]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def read_checksums(path: str | Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        result[relative] = digest
    return result


def validate_checksums(bundle_dir: str | Path) -> list[str]:
    root = Path(bundle_dir)
    checksum_file = root / "checksums.sha256"
    if not checksum_file.is_file():
        return ["Missing checksums.sha256"]
    expected = read_checksums(checksum_file)
    errors: list[str] = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing checksummed file: {relative}")
            continue
        actual = sha256_file(path)
        if actual != digest:
            errors.append(f"Checksum mismatch: {relative}")
    extras = set(build_checksums(root)) - set(expected)
    for relative in sorted(extras):
        errors.append(f"Unchecksummed file: {relative}")
    return errors
