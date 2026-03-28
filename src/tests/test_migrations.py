import pytest

from memask.db.migrate import current_version, migrate_down, migrate_up


class TestMigrateUp:
    def test_applies_all_migrations(self, raw_conn):
        migrate_up(raw_conn)
        assert current_version(raw_conn) == 2, "should apply all migrations from scratch"

    def test_is_idempotent(self, raw_conn):
        migrate_up(raw_conn)
        migrate_up(raw_conn)
        assert current_version(raw_conn) == 2, "running migrate_up twice should not fail"

    def test_partial_migration(self, raw_conn):
        migrate_up(raw_conn, target=1)
        assert current_version(raw_conn) == 1, "should stop at target version"

    def test_resumes_from_current(self, raw_conn):
        migrate_up(raw_conn, target=1)
        migrate_up(raw_conn)
        assert current_version(raw_conn) == 2, "should apply remaining migrations"


class TestMigrateDown:
    def test_rollback_all(self, raw_conn):
        migrate_up(raw_conn)
        migrate_down(raw_conn, target=0)
        assert current_version(raw_conn) == 0, "should rollback all migrations"

    def test_rollback_partial(self, raw_conn):
        migrate_up(raw_conn)
        migrate_down(raw_conn, target=1)
        assert current_version(raw_conn) == 1, "should rollback to target version"

    def test_rollback_then_reapply(self, raw_conn):
        migrate_up(raw_conn)
        migrate_down(raw_conn, target=0)
        migrate_up(raw_conn)
        assert current_version(raw_conn) == 2, "should reapply after full rollback"


class TestSchemaIntegrity:
    @pytest.mark.parametrize("table", ["items", "jobs"])
    def test_table_exists_after_migration(self, raw_conn, table):
        migrate_up(raw_conn)
        row = raw_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        assert row is not None, f"{table} table should exist after migration"

    @pytest.mark.parametrize("table", ["items", "jobs"])
    def test_table_removed_after_rollback(self, raw_conn, table):
        migrate_up(raw_conn)
        migrate_down(raw_conn, target=0)
        row = raw_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        assert row is None, f"{table} table should not exist after full rollback"

    def test_items_table_columns(self, raw_conn):
        migrate_up(raw_conn)
        columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(items)").fetchall()}
        expected = {
            "id",
            "type",
            "content",
            "title",
            "status",
            "priority",
            "due_date",
            "category",
            "source",
            "tags",
            "created_at",
            "updated_at",
            "deleted_at",
            "metadata",
        }
        assert columns == expected, "items table should have all expected columns"
