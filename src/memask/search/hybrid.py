import sqlite3

from memask.search.keyword import SearchResult, keyword_search
from memask.search.semantic import semantic_search
from memask.search.vector_store import VectorStore


def hybrid_search(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service,
    query: str,
    *,
    type: str | None = None,
    status: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keyword_weight: float = 0.4,
    semantic_weight: float = 0.6,
    limit: int = 20,
) -> list[SearchResult]:
    kw_results = keyword_search(
        conn,
        query,
        type=type,
        status=status,
        category=category,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )

    sem_results = semantic_search(
        conn,
        vector_store,
        embedding_service,
        query,
        limit=limit,
    )
    sem_results = _apply_filters(
        sem_results,
        type=type,
        category=category,
        status=status,
    )

    return _merge(
        kw_results,
        sem_results,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        limit=limit,
    )


def merge_results(
    kw_results: list[SearchResult],
    sem_results: list[SearchResult],
    *,
    keyword_weight: float = 0.4,
    semantic_weight: float = 0.6,
    limit: int = 20,
) -> list[SearchResult]:
    return _merge(
        kw_results,
        sem_results,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        limit=limit,
    )


def _merge(
    kw_results: list[SearchResult],
    sem_results: list[SearchResult],
    *,
    keyword_weight: float,
    semantic_weight: float,
    limit: int,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    items: dict[str, SearchResult] = {}

    kw_max = max((r.score for r in kw_results), default=0.0)
    kw_ids = set()
    for r in kw_results:
        norm = _normalize(r.score, kw_max)
        scores[r.item.id] = keyword_weight * norm
        items[r.item.id] = r
        kw_ids.add(r.item.id)

    sem_ids = set()
    sem_max = max((r.score for r in sem_results), default=0.0)
    for r in sem_results:
        norm = _normalize(r.score, sem_max)
        scores[r.item.id] = scores.get(r.item.id, 0.0) + semantic_weight * norm
        sem_ids.add(r.item.id)
        if r.item.id not in items:
            items[r.item.id] = r

    both = kw_ids & sem_ids

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:limit]
    return [
        SearchResult(
            item=items[item_id].item,
            score=score,
            source="hybrid" if item_id in both else items[item_id].source,
        )
        for item_id, score in ranked
    ]


def _normalize(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return value / max_value


def _apply_filters(
    results: list[SearchResult],
    *,
    type: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> list[SearchResult]:
    filtered = results
    if type:
        filtered = [r for r in filtered if r.item.type == type]
    if category:
        filtered = [r for r in filtered if r.item.category == category]
    if status:
        filtered = [r for r in filtered if r.item.status == status]
    return filtered
