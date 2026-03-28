import importlib
import pkgutil
import sqlite3
from types import ModuleType

from memask.clock import now


def _ensure_schema_versions(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
    """)
    conn.commit()


def _discover_migrations() -> list[ModuleType]:
    from memask.db import migrations

    result = []
    for _, name, _ in pkgutil.iter_modules(migrations.__path__):
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"memask.db.migrations.{name}")
        result.append(module)
    result.sort(key=lambda m: m.version)
    return result


def current_version(conn: sqlite3.Connection) -> int:
    _ensure_schema_versions(conn)
    row = conn.execute("SELECT MAX(version) FROM schema_versions").fetchone()
    return row[0] or 0


def migrate_up(conn: sqlite3.Connection, target: int | None = None) -> int:
    _ensure_schema_versions(conn)
    current = current_version(conn)

    for migration in _discover_migrations():
        if migration.version <= current:
            continue
        if target is not None and migration.version > target:
            break
        migration.up(conn)
        conn.execute(
            "INSERT INTO schema_versions (version, name, applied_at) VALUES (?, ?, ?)",
            (migration.version, migration.name, now()),
        )
        conn.commit()

    return current_version(conn)


def migrate_down(conn: sqlite3.Connection, target: int = 0) -> int:
    _ensure_schema_versions(conn)
    current = current_version(conn)

    for migration in reversed(_discover_migrations()):
        if migration.version <= target:
            break
        if migration.version > current:
            continue
        migration.down(conn)
        conn.execute(
            "DELETE FROM schema_versions WHERE version = ?",
            (migration.version,),
        )
        conn.commit()

    return current_version(conn)
