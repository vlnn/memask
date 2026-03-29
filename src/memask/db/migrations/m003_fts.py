import sqlite3

version = 3
name = "add_fts5_index"


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE VIRTUAL TABLE items_fts USING fts5(
            title, content, tags,
            content='items',
            content_rowid='rowid'
        )
    """)
    conn.execute("""
        INSERT INTO items_fts(rowid, title, content, tags)
        SELECT rowid, title, content, tags FROM items WHERE deleted_at IS NULL
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_insert AFTER INSERT ON items BEGIN
            INSERT INTO items_fts(rowid, title, content, tags)
            VALUES (new.rowid, new.title, new.content, new.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_delete AFTER DELETE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, content, tags)
            VALUES ('delete', old.rowid, old.title, old.content, old.tags);
        END
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_update AFTER UPDATE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, content, tags)
            VALUES ('delete', old.rowid, old.title, old.content, old.tags);
            INSERT INTO items_fts(rowid, title, content, tags)
            VALUES (new.rowid, new.title, new.content, new.tags);
        END
    """)


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS items_fts_update")
    conn.execute("DROP TRIGGER IF EXISTS items_fts_delete")
    conn.execute("DROP TRIGGER IF EXISTS items_fts_insert")
    conn.execute("DROP TABLE IF EXISTS items_fts")
