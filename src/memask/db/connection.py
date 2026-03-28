import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".memask" / "memask.db"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    if db_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = Path(db_path) if db_path else DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
