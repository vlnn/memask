import sqlite3

from memask.ulid import ulid

from memask.clock import now
from memask.models.item import Item

VALID_TYPES = {"note", "todo", "url", "decision", "guide"}
UPDATABLE_FIELDS = {
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


def create_item(
    conn: sqlite3.Connection,
    content: str,
    type: str = "note",
    **kwargs: str | int | None,
) -> Item:
    if type not in VALID_TYPES:
        raise ValueError(f"Invalid type: {type}. Must be one of {VALID_TYPES}")

    item_id = str(ulid())
    timestamp = now()

    fields = {
        "id": item_id,
        "type": type,
        "content": content,
        "created_at": timestamp,
        "updated_at": timestamp,
        **{k: v for k, v in kwargs.items() if k in UPDATABLE_FIELDS},
    }

    columns = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)

    conn.execute(
        f"INSERT INTO items ({columns}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()
    return get_item(conn, item_id)


def get_item(conn: sqlite3.Connection, item_id: str) -> Item | None:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
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
        "UPDATE items SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
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
    include_deleted: bool = False,
) -> list[Item]:
    conditions: list[str] = []
    params: list[str] = []

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

    query = "SELECT * FROM items"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()
    return [Item.from_row(row) for row in rows]
