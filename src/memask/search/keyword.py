import sqlite3
from dataclasses import dataclass

from memask.models.item import Item

ITEM_COLUMNS = (
    "items.id, items.type, items.content, items.title, items.status, "
    "items.priority, items.due_date, items.category, items.source, "
    "items.tags, items.created_at, items.updated_at, items.deleted_at, "
    "items.metadata"
)


@dataclass(frozen=True)
class SearchResult:
    item: Item
    score: float
    source: str = "keyword"


def _build_fts_query(text: str) -> str:
    words = text.strip().split()
    if not words:
        return ""
    return " ".join(f'"{w.replace(chr(34), chr(34) + chr(34))}"' for w in words)


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    type: str | None = None,
    status: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    if not query or not query.strip():
        return []

    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    sql = f"""
        SELECT {ITEM_COLUMNS}, bm25(items_fts) AS rank
        FROM items_fts
        JOIN items ON items.rowid = items_fts.rowid
        WHERE items_fts MATCH ?
          AND items.deleted_at IS NULL
    """
    params: list = [fts_query]

    if type:
        sql += " AND items.type = ?"
        params.append(type)
    if status:
        sql += " AND items.status = ?"
        params.append(status)
    if category:
        sql += " AND items.category = ?"
        params.append(category)
    if date_from:
        sql += " AND items.created_at >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND items.created_at <= ?"
        params.append(date_to)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(item=Item.from_row(row), score=abs(row["rank"]), source="keyword")
        for row in rows
    ]
