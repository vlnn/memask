import sqlite3

version = 3
name = "create_items_fts"


def up(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE VIRTUAL TABLE items_fts USING fts5(
            title,
            content,
            content='items',
            content_rowid='rowid'
        )
    """)
    conn.execute("""
        INSERT INTO items_fts(rowid, title, content)
        SELECT rowid, COALESCE(title, ''), content FROM items
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_insert AFTER INSERT ON items BEGIN
            INSERT INTO items_fts(rowid, title, content)
            VALUES (new.rowid, COALESCE(new.title, ''), new.content);
        END
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_delete AFTER DELETE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, content)
            VALUES ('delete', old.rowid, COALESCE(old.title, ''), old.content);
        END
    """)
    conn.execute("""
        CREATE TRIGGER items_fts_update AFTER UPDATE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, title, content)
            VALUES ('delete', old.rowid, COALESCE(old.title, ''), old.content);
            INSERT INTO items_fts(rowid, title, content)
            VALUES (new.rowid, COALESCE(new.title, ''), new.content);
        END
    """)


def down(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER IF EXISTS items_fts_insert")
    conn.execute("DROP TRIGGER IF EXISTS items_fts_delete")
    conn.execute("DROP TRIGGER IF EXISTS items_fts_update")
    conn.execute("DROP TABLE IF EXISTS items_fts")
