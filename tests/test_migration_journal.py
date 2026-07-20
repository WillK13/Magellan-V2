from datetime import datetime, timezone

from magellan.migration.journal import MigrationJournal
from magellan.migration.models import (
    MigrationRecord,
    MigrationRole,
    MigrationStatus,
)


def test_migration_journal_is_atomic_and_persistent(tmp_path) -> None:
    journal = MigrationJournal(tmp_path)
    record = MigrationRecord(
        migration_id="migration-1",
        bid_id="bid-1",
        task_id="task-1",
        source_node_id="boston",
        destination_node_id="virginia",
        generation=1,
        migration_at_utc=datetime.now(timezone.utc),
        role=MigrationRole.SOURCE,
        status=MigrationStatus.ACTIVATING,
    )
    journal.put(record)

    restarted = MigrationJournal(tmp_path)
    loaded = restarted.get(record.migration_id)

    assert loaded == record
    assert len(restarted.list_records()) == 1
