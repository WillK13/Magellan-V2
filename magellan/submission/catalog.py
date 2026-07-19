from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from magellan.submission.models import (
    TaskCatalogSnapshot,
    TaskDefinitionRecord,
    TaskDefinitionSubmission,
    TaskRunRecord,
    TaskRunSubmission,
    canonical_digest,
)


class TaskCatalogStore:
    """Atomic durable catalog for immutable definitions and task runs."""

    def __init__(self, state_root: str | Path, local_node_id: str) -> None:
        self._path = Path(state_root) / "control" / "task_catalog.json"
        self._local_node_id = local_node_id
        self._lock = RLock()
        self._definitions: dict[tuple[str, int], TaskDefinitionRecord] = {}
        self._runs: dict[str, TaskRunRecord] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for value in raw.get("definitions", []):
            record = TaskDefinitionRecord.model_validate(value)
            self._definitions[(record.definition_id, record.revision)] = record
        for value in raw.get("runs", []):
            record = TaskRunRecord.model_validate(value)
            self._runs[record.run_id] = record

    def _persist(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "definitions": [
                        item.model_dump(mode="json")
                        for item in self.list_definitions()
                    ],
                    "runs": [
                        item.model_dump(mode="json")
                        for item in self.list_runs()
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self._path)

    def list_definitions(self) -> list[TaskDefinitionRecord]:
        return [
            self._definitions[key].model_copy(deep=True)
            for key in sorted(self._definitions)
        ]

    def list_runs(self) -> list[TaskRunRecord]:
        return [
            self._runs[key].model_copy(deep=True)
            for key in sorted(self._runs)
        ]

    def get_definition(
        self,
        definition_id: str,
        revision: int | None = None,
    ) -> TaskDefinitionRecord:
        candidates = [
            item
            for (item_id, _), item in self._definitions.items()
            if item_id == definition_id
        ]
        if not candidates:
            raise KeyError(f"Unknown task definition: {definition_id}")
        if revision is None:
            return max(candidates, key=lambda item: item.revision).model_copy(deep=True)
        try:
            return self._definitions[(definition_id, revision)].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(
                f"Unknown task definition revision: {definition_id}@{revision}"
            ) from exc

    def get_run(self, run_id: str) -> TaskRunRecord:
        try:
            return self._runs[run_id].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError(f"Unknown task run: {run_id}") from exc

    def register_definition(
        self,
        submission: TaskDefinitionSubmission,
        origin_node_id: str | None = None,
    ) -> tuple[TaskDefinitionRecord, bool]:
        with self._lock:
            digest = canonical_digest(submission)
            existing = [
                item
                for item in self._definitions.values()
                if item.definition_id == submission.definition_id
            ]
            for item in existing:
                if item.digest == digest:
                    return item.model_copy(deep=True), False
            revision = max((item.revision for item in existing), default=0) + 1
            record = TaskDefinitionRecord(
                **submission.model_dump(),
                revision=revision,
                digest=digest,
                origin_node_id=origin_node_id or self._local_node_id,
            )
            self._definitions[(record.definition_id, record.revision)] = record
            self._persist()
            return record.model_copy(deep=True), True

    def create_run(
        self,
        submission: TaskRunSubmission,
        owner_node_id: str,
        origin_node_id: str | None = None,
    ) -> tuple[TaskRunRecord, bool]:
        with self._lock:
            request_digest = canonical_digest(submission)
            for item in self._runs.values():
                if item.idempotency_key != submission.idempotency_key:
                    continue
                if item.request_digest != request_digest:
                    raise ValueError(
                        "Idempotency key was already used with a different request"
                    )
                return item.model_copy(deep=True), False

            definition = self.get_definition(
                submission.definition_id,
                submission.revision,
            )
            record = TaskRunRecord(
                run_id=f"run-{uuid4()}",
                definition_id=definition.definition_id,
                revision=definition.revision,
                definition_digest=definition.digest,
                origin_node_id=origin_node_id or self._local_node_id,
                initial_owner_node_id=owner_node_id,
                idempotency_key=submission.idempotency_key,
                request_digest=request_digest,
                auto_start=submission.auto_start,
                labels=dict(submission.labels),
            )
            self._runs[record.run_id] = record
            self._persist()
            return record.model_copy(deep=True), True

    def materialize_run(self, run: TaskRunRecord):
        definition = self.get_definition(run.definition_id, run.revision)
        if definition.digest != run.definition_digest:
            raise RuntimeError(
                f"Definition digest mismatch for task run {run.run_id}"
            )
        return definition.materialize(run.run_id, run.initial_owner_node_id)

    def materialized_definitions(self) -> list:
        return [self.materialize_run(run) for run in self.list_runs()]

    def snapshot(self) -> TaskCatalogSnapshot:
        return TaskCatalogSnapshot(
            reporting_node_id=self._local_node_id,
            definitions=self.list_definitions(),
            runs=self.list_runs(),
        )

    def merge_snapshot(
        self,
        snapshot: TaskCatalogSnapshot,
    ) -> tuple[list[TaskDefinitionRecord], list[TaskRunRecord]]:
        added_definitions: list[TaskDefinitionRecord] = []
        added_runs: list[TaskRunRecord] = []
        with self._lock:
            for incoming in snapshot.definitions:
                key = (incoming.definition_id, incoming.revision)
                local = self._definitions.get(key)
                if local is not None:
                    if local.digest != incoming.digest:
                        raise RuntimeError(
                            "Conflicting immutable task definition "
                            f"{incoming.definition_id}@{incoming.revision}"
                        )
                    continue
                self._definitions[key] = incoming.model_copy(deep=True)
                added_definitions.append(incoming.model_copy(deep=True))

            for incoming in snapshot.runs:
                for existing in self._runs.values():
                    if (
                        existing.idempotency_key == incoming.idempotency_key
                        and existing.run_id != incoming.run_id
                    ):
                        raise RuntimeError(
                            "Conflicting task runs use idempotency key "
                            f"{incoming.idempotency_key}"
                        )
                local = self._runs.get(incoming.run_id)
                if local is not None:
                    if local.model_dump(mode="json") != incoming.model_dump(mode="json"):
                        raise RuntimeError(
                            f"Conflicting immutable task run {incoming.run_id}"
                        )
                    continue
                # Definitions are merged before runs, so this validates the reference.
                definition = self.get_definition(
                    incoming.definition_id,
                    incoming.revision,
                )
                if definition.digest != incoming.definition_digest:
                    raise RuntimeError(
                        f"Definition digest mismatch for replicated run {incoming.run_id}"
                    )
                self._runs[incoming.run_id] = incoming.model_copy(deep=True)
                added_runs.append(incoming.model_copy(deep=True))

            if added_definitions or added_runs:
                self._persist()
        return added_definitions, added_runs
