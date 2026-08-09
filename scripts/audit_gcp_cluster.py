#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass

from magellan.config.loader import load_cluster_config


@dataclass
class AuditResult:
    node_id: str
    configured_vm_name: str
    actual_vm_name: str | None
    configured_zone: str
    configured_internal_ip: str
    actual_zone: str | None
    actual_internal_ip: str | None
    status: str | None
    machine_type: str | None
    matches: bool
    matched_by: str | None
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare config/cluster.gcp.json with live GCP VM metadata. "
            "Instances are matched by configured name first and internal IP second, "
            "so stale VM names can be discovered."
        )
    )
    parser.add_argument("--cluster", default="config/cluster.gcp.json")
    parser.add_argument("--project", default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def _tail(value: str | None) -> str | None:
    if value is None:
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def _instance_ip(payload: dict) -> str | None:
    interfaces = payload.get("networkInterfaces") or []
    if not interfaces:
        return None
    return interfaces[0].get("networkIP")


def _list_instances(project: str | None) -> list[dict]:
    command = [
        "gcloud",
        "compute",
        "instances",
        "list",
        "--format=json",
    ]
    if project:
        command.extend(["--project", project])
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    args = parse_args()
    cluster = load_cluster_config(args.cluster)

    try:
        instances = _list_instances(args.project)
    except FileNotFoundError:
        raise SystemExit("gcloud executable was not found")
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise SystemExit(f"gcloud instance listing failed: {message}") from exc

    results: list[AuditResult] = []
    for node in cluster.nodes:
        configured_ip = str(node.internal_ip)
        exact = [
            item
            for item in instances
            if item.get("name") == node.vm_name and _tail(item.get("zone")) == node.zone
        ]
        ip_matches = [item for item in instances if _instance_ip(item) == configured_ip]

        match = exact[0] if exact else (ip_matches[0] if len(ip_matches) == 1 else None)
        matched_by = "name+zone" if exact else ("internal_ip" if match else None)
        errors: list[str] = []

        if match is None:
            if len(ip_matches) > 1:
                errors.append(
                    f"multiple live GCP instances unexpectedly use internal IP {configured_ip}"
                )
            else:
                errors.append(
                    "no live GCP instance matched configured name+zone or internal IP"
                )
            actual_name = None
            actual_zone = None
            actual_ip = None
            status = None
            machine_type = None
        else:
            actual_name = match.get("name")
            actual_zone = _tail(match.get("zone"))
            actual_ip = _instance_ip(match)
            status = match.get("status")
            machine_type = _tail(match.get("machineType"))

            if actual_name != node.vm_name:
                errors.append(
                    f"VM name mismatch: config={node.vm_name!r}, GCP={actual_name!r}"
                )
            if actual_zone != node.zone:
                errors.append(
                    f"zone mismatch: config={node.zone}, GCP={actual_zone}"
                )
            if actual_ip != configured_ip:
                errors.append(
                    f"internal IP mismatch: config={configured_ip}, GCP={actual_ip}"
                )
            if status != "RUNNING":
                errors.append(f"instance status is {status!r}, expected RUNNING")

        results.append(
            AuditResult(
                node_id=node.id,
                configured_vm_name=node.vm_name,
                actual_vm_name=actual_name,
                configured_zone=node.zone,
                configured_internal_ip=configured_ip,
                actual_zone=actual_zone,
                actual_internal_ip=actual_ip,
                status=status,
                machine_type=machine_type,
                matches=not errors,
                matched_by=matched_by,
                errors=errors,
            )
        )

    valid = all(item.matches for item in results)
    if args.as_json:
        print(
            json.dumps(
                {
                    "valid": valid,
                    "results": [asdict(item) for item in results],
                },
                indent=2,
            )
        )
    else:
        print("== Live GCP cluster audit ==")
        for item in results:
            marker = "OK" if item.matches else "FAIL"
            print(
                f"[{marker}] {item.node_id:16s} "
                f"configured={item.configured_vm_name:30s} "
                f"actual={item.actual_vm_name or '-':30s} "
                f"zone={item.actual_zone or '-':24s} "
                f"ip={item.actual_internal_ip or '-':15s} "
                f"status={item.status or '-':12s} "
                f"type={item.machine_type or '-'}"
            )
            for error in item.errors:
                print(f"       {error}")
        print("GCP CLUSTER AUDIT PASSED" if valid else "GCP CLUSTER AUDIT FAILED")

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
