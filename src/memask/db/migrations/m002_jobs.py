import sqlite3

version = 2
name = "create_jobs_table"


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE jobs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_jobs_status ON jobs (status)")
    conn.execute("CREATE INDEX idx_jobs_type ON jobs (type)")


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS jobs")
