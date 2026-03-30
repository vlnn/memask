import sqlite3

from memask.clock import now
from memask.models.item import Item
from memask.ulid import ulid

UPDATABLE_FIELDS = frozenset(
    {
        "type",
        "content",
        "title",
        "status",
        "priority",
        "due_date",
        "category",
        "source",
        "tags",
        "metadata",
    }
)


VALID_TYPES = frozenset(
    {
        "note",
        "todo",
        "url",
        "decision",
        "guide",
    }
)


def create_item(
    conn: sqlite3.Connection,
    content: str,
    *,
    type: str = "note",
    title: str | None = None,
    status: str | None = None,
    priority: int | None = None,
    due_date: str | None = None,
    category: str | None = None,
    source: str = "manual",
    tags: str | None = None,
    metadata: str | None = None,
    **_kwargs,
) -> Item:
    if type not in VALID_TYPES:
        raise ValueError(f"Invalid type: {type}")
    item_id = ulid()
    timestamp = now()

    conn.execute(
        """INSERT INTO items
           (id, type, content, title, status, priority,
            due_date, category, source, tags,
            created_at, updated_at, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            item_id,
            type,
            content,
            title,
            status,
            priority,
            due_date,
            category,
            source,
            tags,
            timestamp,
            timestamp,
            metadata,
        ),
    )
    conn.commit()
    return get_item(conn, item_id)


def get_item(conn: sqlite3.Connection, item_id: str) -> Item | None:
    row = conn.execute(
        "SELECT * FROM items WHERE id = ?",
        (item_id,),
    ).fetchone()
    return Item.from_row(row) if row else None


def update_item(
    conn: sqlite3.Connection,
    item_id: str,
    **fields: str | int | None,
) -> Item | None:
    updates = {k: v for k, v in fields.items() if k in UPDATABLE_FIELDS}
    if not updates:
        return get_item(conn, item_id)

    updates["updated_at"] = now()
    set_clause = ", ".join(f"{k} = ?" for k in updates)

    conn.execute(
        f"UPDATE items SET {set_clause} WHERE id = ? AND deleted_at IS NULL",
        (*updates.values(), item_id),
    )
    conn.commit()
    return get_item(conn, item_id)


def soft_delete_item(conn: sqlite3.Connection, item_id: str) -> bool:
    timestamp = now()
    cursor = conn.execute(
        "UPDATE items SET deleted_at = ?, updated_at = ?"
        " WHERE id = ? AND deleted_at IS NULL",
        (timestamp, timestamp, item_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def list_items(
    conn: sqlite3.Connection,
    *,
    type: str | None = None,
    status: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_deleted: bool = False,
    limit: int = 100,
) -> list[Item]:
    conditions = []
    params: list = []

    if not include_deleted:
        conditions.append("deleted_at IS NULL")
    if type:
        conditions.append("type = ?")
        params.append(type)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if category:
        conditions.append("category = ?")
        params.append(category)
    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("created_at <= ?")
        params.append(date_to)

    where = " AND ".join(conditions) if conditions else "1=1"
    rows = conn.execute(
        f"SELECT * FROM items WHERE {where} ORDER BY created_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [Item.from_row(r) for r in rows]
