import pytest

from memask.repository.items import create_item
from memask.search.indexer import index_item
from memask.search.semantic import semantic_search
from memask.search.vector_store import VectorStore


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


class TestSemanticSearch:
    def test_finds_indexed_item(self, conn, store, fake_embedder):
        item = create_item(conn, "deploy to production")
        index_item(conn, store, fake_embedder, item.id)

        results = semantic_search(conn, store, fake_embedder, "deploy to production")
        assert len(results) >= 1, "should find the indexed item"
        assert results[0].item.id == item.id, "should return the correct item"
        assert results[0].source == "semantic", "should tag source as semantic"

    def test_returns_empty_for_blank_query(self, conn, store, fake_embedder):
        results = semantic_search(conn, store, fake_embedder, "   ")
        assert results == [], "should return empty for blank query"

    def test_excludes_soft_deleted(self, conn, store, fake_embedder):
        item = create_item(conn, "secret note")
        index_item(conn, store, fake_embedder, item.id)
        conn.execute(
            "UPDATE items SET deleted_at = '2025-01-01' WHERE id = ?", (item.id,)
        )
        conn.commit()

        results = semantic_search(conn, store, fake_embedder, "secret note")
        assert results == [], "should not return soft-deleted items"

    def test_deduplicates_chunked_items(self, conn, store, fake_embedder):
        content = "word " * 300
        item = create_item(conn, content)
        index_item(conn, store, fake_embedder, item.id)

        results = semantic_search(conn, store, fake_embedder, "word")
        item_ids = [r.item.id for r in results]
        assert item_ids.count(item.id) == 1, "each item should appear at most once"

    def test_scores_are_non_negative(self, conn, store, fake_embedder):
        item = create_item(conn, "test content")
        index_item(conn, store, fake_embedder, item.id)

        results = semantic_search(conn, store, fake_embedder, "test content")
        for r in results:
            assert r.score >= 0, "similarity score should be non-negative"

    def test_multiple_items_ranked(self, conn, store, fake_embedder):
        for i in range(5):
            item = create_item(conn, f"topic number {i} about different things")
            index_item(conn, store, fake_embedder, item.id)

        results = semantic_search(conn, store, fake_embedder, "topic", limit=5)
        assert len(results) == 5, "should return all indexed items"
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "results should be sorted by score descending"
