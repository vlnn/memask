import pytest

from memask.repository.items import create_item, soft_delete_item
from memask.search.indexer import index_item, reconcile, reindex_stale
from memask.search.vector_store import VectorStore


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


class TestIndexItem:
    def test_indexes_short_item_as_single_vector(self, conn, store, fake_embedder):
        item = create_item(conn, "buy milk")
        count = index_item(conn, store, fake_embedder, item.id)
        assert count == 1, "short item should produce one vector"
        assert store.count() == 1, "store should have one record"

    def test_indexes_long_item_as_chunks(self, conn, store, fake_embedder):
        content = "word " * 300
        item = create_item(conn, content)
        count = index_item(conn, store, fake_embedder, item.id)
        assert count > 1, "long item should produce multiple chunks"
        assert store.count() == count, "store should have one record per chunk"

    def test_includes_title_in_text(self, conn, store, fake_embedder):
        item = create_item(conn, "some content", title="Deployment Guide")
        index_item(conn, store, fake_embedder, item.id)
        calls = fake_embedder.call_log
        embedded_text = calls[0][1]
        assert "Deployment Guide" in embedded_text, "embedded text should include title"
        assert "some content" in embedded_text, "embedded text should include content"

    def test_skips_deleted_item(self, conn, store, fake_embedder):
        item = create_item(conn, "will be deleted")
        soft_delete_item(conn, item.id)
        count = index_item(conn, store, fake_embedder, item.id)
        assert count == 0, "should not index a deleted item"

    def test_skips_nonexistent_item(self, conn, store, fake_embedder):
        count = index_item(conn, store, fake_embedder, "no-such-id")
        assert count == 0, "should return 0 for nonexistent item"

    def test_reindex_replaces_old_vectors(self, conn, store, fake_embedder):
        item = create_item(conn, "original content")
        index_item(conn, store, fake_embedder, item.id)
        assert store.count() == 1, "should have one vector initially"

        conn.execute(
            "UPDATE items SET content = ? WHERE id = ?",
            ("updated content that is much longer " * 30, item.id),
        )
        conn.commit()

        count = index_item(conn, store, fake_embedder, item.id)
        assert count >= 1, "should reindex successfully"
        ids = store.list_item_ids()
        assert ids == {item.id}, "should only have vectors for this item"


class TestReconcile:
    def test_removes_orphaned_vectors(self, conn, store, fake_embedder):
        item = create_item(conn, "will be deleted")
        index_item(conn, store, fake_embedder, item.id)
        assert store.count() == 1, "should have one vector before delete"

        soft_delete_item(conn, item.id)
        removed = reconcile(conn, store)
        assert removed == 1, "should remove one orphaned item"
        assert store.count() == 0, "store should be empty after reconciliation"

    def test_keeps_active_items(self, conn, store, fake_embedder):
        item = create_item(conn, "active note")
        index_item(conn, store, fake_embedder, item.id)

        removed = reconcile(conn, store)
        assert removed == 0, "should not remove active items"
        assert store.count() == 1, "active item should remain in store"

    def test_mixed_active_and_orphaned(self, conn, store, fake_embedder):
        active = create_item(conn, "keep this")
        orphan = create_item(conn, "delete this")
        index_item(conn, store, fake_embedder, active.id)
        index_item(conn, store, fake_embedder, orphan.id)

        soft_delete_item(conn, orphan.id)
        removed = reconcile(conn, store)
        assert removed == 1, "should remove only the orphaned item"
        assert store.list_item_ids() == {active.id}, "only active item should remain"


class TestReindexStale:
    def test_reindexes_items_with_old_model(self, conn, store, fake_embedder):
        from conftest import FakeEmbeddingService

        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = create_item(conn, "stale content")
        index_item(conn, store, old_embedder, item.id)

        stale = store.stale_items(fake_embedder.model_name)
        assert item.id in stale, "item should be stale under new model"

        count = reindex_stale(conn, store, fake_embedder)
        assert count >= 1, "should reindex at least one item"

        stale_after = store.stale_items(fake_embedder.model_name)
        assert item.id not in stale_after, "item should no longer be stale"

    def test_nothing_stale(self, conn, store, fake_embedder):
        item = create_item(conn, "fresh content")
        index_item(conn, store, fake_embedder, item.id)

        count = reindex_stale(conn, store, fake_embedder)
        assert count == 0, "should reindex nothing when all items are current"
