import sqlite3

version = 4
name = "create_instructions_table"


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE instructions (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_instructions_active ON instructions (active)")


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS instructions")
