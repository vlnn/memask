from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memask.search.keyword import SearchResult

if TYPE_CHECKING:
    from memask.context import Reranker

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(
    query: str,
    results: list[SearchResult],
    reranker: Reranker | None,
    *,
    top_n: int = 10,
) -> list[SearchResult]:
    if not results:
        return []

    if reranker is None:
        return results[:top_n]

    pairs = [(r.item.content, query) for r in results]
    scores = reranker.score_pairs(pairs)

    scored = [
        SearchResult(item=r.item, score=s, source="reranked")
        for r, s in zip(results, scores)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load_model()
        raw_scores = model.predict(pairs)
        return [float(s) for s in raw_scores]

    def _load_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder
        logger.info("loading cross-encoder model: %s", self._model_name)
        self._model = CrossEncoder(self._model_name)
        return self._model
