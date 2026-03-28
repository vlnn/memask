import pytest

from memask.models.item import Item
from memask.repository.items import create_item
from memask.search.hybrid import merge_results, hybrid_search
from memask.search.indexer import index_item
from memask.search.keyword import SearchResult
from memask.search.vector_store import VectorStore


def _fake_item(id: str, content: str = "test") -> Item:
    return Item(
        id=id, type="note", content=content, title=None,
        status=None, priority=None, due_date=None, category=None,
        source="manual", tags=None, created_at="2025-01-01",
        updated_at="2025-01-01", deleted_at=None, metadata=None,
    )


class TestMergeResults:
    def test_merges_disjoint_results(self):
        kw = [SearchResult(item=_fake_item("a"), score=1.0, source="keyword")]
        sem = [SearchResult(item=_fake_item("b"), score=1.0, source="semantic")]

        merged = merge_results(kw, sem, keyword_weight=0.4, semantic_weight=0.6)
        ids = [r.item.id for r in merged]
        assert "a" in ids, "keyword result should be in merged output"
        assert "b" in ids, "semantic result should be in merged output"

    def test_deduplicates_shared_items(self):
        kw = [SearchResult(item=_fake_item("shared"), score=1.0, source="keyword")]
        sem = [SearchResult(item=_fake_item("shared"), score=1.0, source="semantic")]

        merged = merge_results(kw, sem)
        ids = [r.item.id for r in merged]
        assert ids.count("shared") == 1, "shared item should appear once"
        assert merged[0].source == "hybrid", "shared item should be tagged hybrid"

    def test_combined_score_higher_than_individual(self):
        kw = [SearchResult(item=_fake_item("both"), score=1.0, source="keyword")]
        sem = [SearchResult(item=_fake_item("both"), score=1.0, source="semantic")]
        kw_only = [SearchResult(item=_fake_item("kw-only"), score=1.0, source="keyword")]

        merged = merge_results(kw + kw_only, sem, keyword_weight=0.4, semantic_weight=0.6)
        both_score = next(r.score for r in merged if r.item.id == "both")
        kw_score = next(r.score for r in merged if r.item.id == "kw-only")
        assert both_score > kw_score, "item found by both should score higher"

    def test_respects_weights(self):
        kw = [SearchResult(item=_fake_item("a"), score=1.0, source="keyword")]
        sem = [SearchResult(item=_fake_item("b"), score=1.0, source="semantic")]

        merged_kw_heavy = merge_results(kw, sem, keyword_weight=0.9, semantic_weight=0.1)
        merged_sem_heavy = merge_results(kw, sem, keyword_weight=0.1, semantic_weight=0.9)

        assert merged_kw_heavy[0].item.id == "a", "keyword item should rank first with keyword-heavy weights"
        assert merged_sem_heavy[0].item.id == "b", "semantic item should rank first with semantic-heavy weights"

    def test_normalizes_scores(self):
        kw = [
            SearchResult(item=_fake_item("high"), score=10.0, source="keyword"),
            SearchResult(item=_fake_item("low"), score=1.0, source="keyword"),
        ]
        sem = []

        merged = merge_results(kw, sem, keyword_weight=1.0, semantic_weight=0.0)
        assert merged[0].item.id == "high", "highest score should rank first"
        assert merged[1].item.id == "low", "lowest score should rank last"

    def test_empty_keyword_results(self):
        sem = [SearchResult(item=_fake_item("a"), score=0.8, source="semantic")]
        merged = merge_results([], sem)
        assert len(merged) == 1, "should return semantic results when keyword is empty"

    def test_empty_semantic_results(self):
        kw = [SearchResult(item=_fake_item("a"), score=0.8, source="keyword")]
        merged = merge_results(kw, [])
        assert len(merged) == 1, "should return keyword results when semantic is empty"

    def test_both_empty(self):
        merged = merge_results([], [])
        assert merged == [], "should return empty when both are empty"

    def test_respects_limit(self):
        kw = [SearchResult(item=_fake_item(f"kw-{i}"), score=1.0 - i * 0.1, source="keyword") for i in range(10)]
        merged = merge_results(kw, [], limit=3)
        assert len(merged) == 3, "should respect limit"

    def test_sorted_by_score_descending(self):
        kw = [
            SearchResult(item=_fake_item("c"), score=0.3, source="keyword"),
            SearchResult(item=_fake_item("a"), score=0.9, source="keyword"),
            SearchResult(item=_fake_item("b"), score=0.6, source="keyword"),
        ]
        merged = merge_results(kw, [])
        scores = [r.score for r in merged]
        assert scores == sorted(scores, reverse=True), "results should be sorted by score descending"


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


class TestHybridSearchIntegration:
    def test_combines_keyword_and_semantic(self, conn, store, fake_embedder):
        item = create_item(conn, "deploy to production environment")
        index_item(conn, store, fake_embedder, item.id)

        results = hybrid_search(conn, store, fake_embedder, "deploy production")
        assert len(results) >= 1, "should find the item via hybrid search"

    def test_returns_empty_for_blank_query(self, conn, store, fake_embedder):
        results = hybrid_search(conn, store, fake_embedder, "   ")
        assert results == [], "should return empty for blank query"

    def test_keyword_only_match(self, conn, store, fake_embedder):
        create_item(conn, "unindexed keyword content")
        results = hybrid_search(conn, store, fake_embedder, "unindexed")
        assert len(results) >= 1, "should find item via keyword even if not in vector store"

    def test_filters_applied(self, conn, store, fake_embedder):
        note = create_item(conn, "deploy note", type="note")
        todo = create_item(conn, "deploy todo", type="todo")
        index_item(conn, store, fake_embedder, note.id)
        index_item(conn, store, fake_embedder, todo.id)

        results = hybrid_search(conn, store, fake_embedder, "deploy", type="todo")
        types = {r.item.type for r in results}
        assert types == {"todo"}, "should only return items matching type filter"
