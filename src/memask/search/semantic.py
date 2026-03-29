import sqlite3

from memask.models.item import Item
from memask.search.keyword import SearchResult
from memask.search.vector_store import VectorStore


def semantic_search(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    if not query or not query.strip():
        return []

    query_vector = embedding_service.embed_one(query)
    raw_results = vector_store.search(query_vector, limit=limit)

    results = []
    seen_ids = set()
    for row in raw_results:
        item_id = row["item_id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        item_row = conn.execute(
            "SELECT * FROM items WHERE id = ? AND deleted_at IS NULL",
            (item_id,),
        ).fetchone()

        if item_row:
            score = max(0.0, 1.0 - row["score"])
            results.append(
                SearchResult(
                    item=Item.from_row(item_row), score=score, source="semantic"
                )
            )

    return results
