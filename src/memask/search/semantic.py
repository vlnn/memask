import sqlite3

from memask.models.item import Item
from memask.search.embedding import EmbeddingService
from memask.search.keyword import SearchResult
from memask.search.vector_store import VectorStore


def semantic_search(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    query: str,
    *,
    limit: int = 20,
) -> list[SearchResult]:
    if not query.strip():
        return []

    query_vector = embedding_service.embed_one(query)
    raw_results = vector_store.search(
        query_vector,
        limit=limit,
        model_version=embedding_service.model_name,
    )

    results = []
    seen_ids = set()

    for r in raw_results:
        item_id = r["item_id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        item = _fetch_active_item(conn, item_id)
        if item is None:
            continue

        similarity = _distance_to_similarity(r["distance"])
        results.append(SearchResult(item=item, score=similarity, source="semantic"))

    return results


def _fetch_active_item(conn: sqlite3.Connection, item_id: str) -> Item | None:
    row = conn.execute(
        "SELECT * FROM items WHERE id = ? AND deleted_at IS NULL",
        (item_id,),
    ).fetchone()
    return Item.from_row(row) if row else None


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, 1.0 - distance)
