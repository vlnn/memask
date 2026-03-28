import sqlite3
from dataclasses import dataclass

from memask.models.item import Item


@dataclass(frozen=True)
class SearchResult:
    item: Item
    rank: float


def search_items(
    conn: sqlite3.Connection,
    query: str,
    *,
    type: str | None = None,
    status: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_deleted: bool = False,
    limit: int = 20,
) -> list[SearchResult]:
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    conditions = ["items_fts MATCH ?"]
    params: list[str | int] = [fts_query]

    if not include_deleted:
        conditions.append("items.deleted_at IS NULL")
    if type:
        conditions.append("items.type = ?")
        params.append(type)
    if status:
        conditions.append("items.status = ?")
        params.append(status)
    if category:
        conditions.append("items.category = ?")
        params.append(category)
    if date_from:
        conditions.append("items.created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("items.created_at <= ?")
        params.append(date_to)

    where_clause = " AND ".join(conditions)
    params.append(limit)

    sql = f"""
        SELECT items.*, items_fts.rank
        FROM items_fts
        JOIN items ON items.rowid = items_fts.rowid
        WHERE {where_clause}
        ORDER BY items_fts.rank
        LIMIT ?
    """

    rows = conn.execute(sql, params).fetchall()
    return [_result_from_row(row) for row in rows]


def _build_fts_query(query: str) -> str:
    tokens = query.strip().split()
    if not tokens:
        return ""
    escaped = [_escape_fts_token(t) for t in tokens]
    return " ".join(escaped)


def _escape_fts_token(token: str) -> str:
    cleaned = "".join(c for c in token if c.isalnum() or c == "_")
    if not cleaned:
        return ""
    return f'"{cleaned}"'


def _result_from_row(row: sqlite3.Row) -> SearchResult:
    row_dict = dict(row)
    rank = row_dict.pop("rank")
    return SearchResult(item=Item(**row_dict), rank=rank)
