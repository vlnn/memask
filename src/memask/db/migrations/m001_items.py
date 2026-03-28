import sqlite3

version = 1
name = "create_items_table"


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE items (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL DEFAULT 'note',
            content TEXT NOT NULL,
            title TEXT,
            status TEXT,
            priority INTEGER,
            due_date TEXT,
            category TEXT,
            source TEXT NOT NULL DEFAULT 'manual',
            tags TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT,
            metadata TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_items_type ON items (type)")
    conn.execute("CREATE INDEX idx_items_status ON items (status)")
    conn.execute("CREATE INDEX idx_items_category ON items (category)")
    conn.execute("CREATE INDEX idx_items_created_at ON items (created_at)")
    conn.execute("CREATE INDEX idx_items_deleted_at ON items (deleted_at)")


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS items")
