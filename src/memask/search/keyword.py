import sqlite3
from dataclasses import dataclass

from memask.models.item import Item


@dataclass(frozen=True)
class SearchResult:
    item: Item
    score: float
    source: str


def keyword_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    type: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    if not query.strip():
        return []

    fts_query = _build_fts_query(query)

    item_columns = (
        "items.id, items.type, items.content, items.title, items.status, "
        "items.priority, items.due_date, items.category, items.source, "
        "items.tags, items.created_at, items.updated_at, items.deleted_at, "
        "items.metadata"
    )
    sql = f"""
        SELECT {item_columns}, bm25(items_fts) AS rank
        FROM items_fts
        JOIN items ON items.rowid = items_fts.rowid
        WHERE items_fts MATCH ?
          AND items.deleted_at IS NULL
    """
    params: list = [fts_query]

    if type is not None:
        sql += " AND items.type = ?"
        params.append(type)
    if category is not None:
        sql += " AND items.category = ?"
        params.append(category)
    if status is not None:
        sql += " AND items.status = ?"
        params.append(status)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        SearchResult(
            item=Item.from_row(row),
            score=-row["rank"],
            source="keyword",
        )
        for row in rows
    ]


def _build_fts_query(query: str) -> str:
    tokens = query.strip().split()
    escaped = [token.replace('"', '""') for token in tokens if token]
    return " ".join(f'"{t}"' for t in escaped)
