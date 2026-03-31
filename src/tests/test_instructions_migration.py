import pytest

from memask.db.migrate import current_version, migrate_up, migrate_down


class TestInstructionsMigration:
    def test_migration_creates_instructions_table(self, raw_conn):
        migrate_up(raw_conn)
        row = raw_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='instructions'"
        ).fetchone()
        assert row is not None, (
            "instructions table should exist after migration"
        )

    def test_instructions_table_has_expected_columns(self, raw_conn):
        migrate_up(raw_conn)
        cursor = raw_conn.execute("PRAGMA table_info(instructions)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {"id", "content", "active", "created_at", "updated_at"}
        assert expected.issubset(columns), (
            f"instructions table should have columns {expected}, got {columns}"
        )

    def test_rollback_drops_instructions_table(self, raw_conn):
        migrate_up(raw_conn)
        version_before = current_version(raw_conn)
        migrate_down(raw_conn, target=version_before - 1)
        row = raw_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='instructions'"
        ).fetchone()
        assert row is None, (
            "instructions table should not exist after rollback"
        )
