import sqlite3

from memask.search.embedding import EmbeddingService
from memask.search.keyword import SearchResult, keyword_search
from memask.search.semantic import semantic_search
from memask.search.vector_store import VectorStore

DEFAULT_KEYWORD_WEIGHT = 0.4
DEFAULT_SEMANTIC_WEIGHT = 0.6


def hybrid_search(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    query: str,
    *,
    type: str | None = None,
    category: str | None = None,
    status: str | None = None,
    limit: int = 20,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> list[SearchResult]:
    if not query.strip():
        return []

    kw_results = keyword_search(
        conn, query, type=type, category=category, status=status, limit=limit,
    )
    sem_results = semantic_search(
        conn, vector_store, embedding_service, query, limit=limit,
    )
    sem_results = _apply_filters(sem_results, type=type, category=category, status=status)

    return merge_results(
        kw_results,
        sem_results,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        limit=limit,
    )


def merge_results(
    keyword_results: list[SearchResult],
    semantic_results: list[SearchResult],
    *,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    limit: int = 20,
) -> list[SearchResult]:
    kw_max = _max_score(keyword_results)
    sem_max = _max_score(semantic_results)

    scored: dict[str, tuple[float, SearchResult]] = {}

    for r in keyword_results:
        normalized = _normalize(r.score, kw_max)
        weighted = normalized * keyword_weight
        scored[r.item.id] = (weighted, SearchResult(
            item=r.item,
            score=weighted,
            source="keyword",
        ))

    for r in semantic_results:
        normalized = _normalize(r.score, sem_max)
        weighted = normalized * semantic_weight
        item_id = r.item.id

        if item_id in scored:
            existing_score, existing_result = scored[item_id]
            combined = existing_score + weighted
            scored[item_id] = (combined, SearchResult(
                item=existing_result.item,
                score=combined,
                source="hybrid",
            ))
        else:
            scored[item_id] = (weighted, SearchResult(
                item=r.item,
                score=weighted,
                source="semantic",
            ))

    ranked = sorted(scored.values(), key=lambda x: x[0], reverse=True)
    return [result for _, result in ranked[:limit]]


def _max_score(results: list[SearchResult]) -> float:
    if not results:
        return 1.0
    return max(r.score for r in results) or 1.0


def _normalize(score: float, max_score: float) -> float:
    if max_score == 0:
        return 0.0
    return score / max_score


def _apply_filters(
    results: list[SearchResult],
    *,
    type: str | None = None,
    category: str | None = None,
    status: str | None = None,
) -> list[SearchResult]:
    filtered = results
    if type is not None:
        filtered = [r for r in filtered if r.item.type == type]
    if category is not None:
        filtered = [r for r in filtered if r.item.category == category]
    if status is not None:
        filtered = [r for r in filtered if r.item.status == status]
    return filtered
