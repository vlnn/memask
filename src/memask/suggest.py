import sqlite3

COMMANDS = [
    {"text": "/todo", "description": "Create a new todo"},
    {"text": "/todo list", "description": "List pending todos"},
    {"text": "/todo done", "description": "Complete a todo"},
    {"text": "/done", "description": "Complete a todo (shortcut)"},
    {"text": "/undone", "description": "Reopen a completed todo"},
    {"text": "/list", "description": "List all items"},
    {"text": "/list notes", "description": "List notes only"},
    {"text": "/list todos", "description": "List todos only"},
    {"text": "/search", "description": "Search items"},
    {"text": "/notes", "description": "List notes (shortcut)"},
    {"text": "/todos", "description": "List todos (shortcut)"},
]


def score_command(query: str, command_text: str) -> float:
    q = query.lower().strip()
    if not q:
        return 0.0
    c = command_text.lower()
    if q == c:
        return 1.0
    if c.startswith(q):
        return 0.7 + 0.3 * (len(q) / len(c))
    if q.lstrip("/") and q.lstrip("/") in c:
        return 0.3
    return 0.0


def _search_items(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
) -> list[dict]:
    stripped = query.strip()
    if not stripped:
        return _recent_items(conn, limit)

    safe_q = stripped.replace("%", "").replace("_", "").replace("'", "")
    if not safe_q:
        return _recent_items(conn, limit)

    rows = conn.execute(
        """SELECT id, type, content FROM items
           WHERE deleted_at IS NULL
             AND content LIKE ?
           ORDER BY updated_at DESC
           LIMIT ?""",
        (f"%{safe_q}%", limit),
    ).fetchall()

    return [
        {
            "text": row["content"],
            "kind": "item",
            "id": row["id"],
            "item_type": row["type"],
            "score": _item_score(safe_q, row["content"]),
        }
        for row in rows
    ]


def _recent_items(conn: sqlite3.Connection, limit: int) -> list[dict]:
    rows = conn.execute(
        """SELECT id, type, content FROM items
           WHERE deleted_at IS NULL
           ORDER BY updated_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    return [
        {
            "text": row["content"],
            "kind": "item",
            "id": row["id"],
            "item_type": row["type"],
            "score": 0.1,
        }
        for row in rows
    ]


def _item_score(query: str, content: str) -> float:
    q = query.lower()
    c = content.lower()
    if c == q:
        return 0.9
    if c.startswith(q):
        return 0.6 + 0.2 * (len(q) / max(len(c), 1))
    return 0.3 + 0.2 * (len(q) / max(len(c), 1))


def suggest(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
) -> list[dict]:
    results = []

    for cmd in COMMANDS:
        s = score_command(query, cmd["text"])
        if s > 0:
            results.append({
                "text": cmd["text"],
                "kind": "command",
                "description": cmd["description"],
                "score": s,
            })

    item_limit = max(limit - len(results), 3)
    items = _search_items(conn, query.lstrip("/"), item_limit)
    results.extend(items)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]
