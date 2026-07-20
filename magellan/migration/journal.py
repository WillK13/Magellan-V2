from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

from magellan.migration.models import MigrationRecord


class MigrationJournal:
    """Atomic, durable migration transaction records for one daemon."""

    def __init__(self, state_root: str | Path) -> None:
        self._directory = Path(state_root) / "control" / "migrations"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _path(self, migration_id: str) -> Path:
        return self._directory / f"{migration_id}.json"

    def put(self, record: MigrationRecord) -> MigrationRecord:
        with self._lock:
            path = self._path(record.migration_id)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record.model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
            return record.model_copy(deep=True)

    def get(self, migration_id: str) -> MigrationRecord | None:
        with self._lock:
            path = self._path(migration_id)
            if not path.is_file():
                return None
            return MigrationRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def list_records(self) -> list[MigrationRecord]:
        with self._lock:
            records = [
                MigrationRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                for path in self._directory.glob("*.json")
            ]
        return sorted(records, key=lambda item: item.created_at_utc)
