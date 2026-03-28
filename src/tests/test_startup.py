import json

import pytest

from conftest import FakeEmbeddingService
from memask.repository.items import create_item, soft_delete_item
from memask.repository.jobs import enqueue, queue_status
from memask.search.indexer import index_item
from memask.search.startup import on_startup, enqueue_stale_reindex
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding, process_all_pending


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


class TestEnqueueStaleReindex:
    def test_enqueues_jobs_for_stale_items(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item_a = create_item(conn, "first note")
        item_b = create_item(conn, "second note")
        index_item(conn, store, old_embedder, item_a.id)
        index_item(conn, store, old_embedder, item_b.id)

        count = enqueue_stale_reindex(conn, store, fake_embedder)
        assert count == 2, "should enqueue one job per stale item"

        status = queue_status(conn)
        assert status.get("pending", 0) == 2, "all reindex jobs should be pending"

    def test_skips_current_model_items(self, conn, store, fake_embedder):
        item = create_item(conn, "fresh note")
        index_item(conn, store, fake_embedder, item.id)

        count = enqueue_stale_reindex(conn, store, fake_embedder)
        assert count == 0, "should not enqueue jobs for items with current model"

    def test_enqueued_jobs_actually_reindex(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = create_item(conn, "stale content")
        index_item(conn, store, old_embedder, item.id)

        enqueue_stale_reindex(conn, store, fake_embedder)
        process_all_pending(conn, store, fake_embedder)

        stale = store.stale_items(fake_embedder.model_name)
        assert stale == set(), "no items should be stale after processing reindex jobs"

    def test_does_not_duplicate_existing_pending_jobs(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = create_item(conn, "stale note")
        index_item(conn, store, old_embedder, item.id)

        enqueue_stale_reindex(conn, store, fake_embedder)
        second_count = enqueue_stale_reindex(conn, store, fake_embedder)
        assert second_count == 0, "should not enqueue duplicates for already-queued items"


class TestOnStartup:
    def test_cleans_orphaned_vectors(self, conn, store, fake_embedder):
        item = create_item(conn, "will be deleted")
        index_item(conn, store, fake_embedder, item.id)
        soft_delete_item(conn, item.id)

        result = on_startup(conn, store, fake_embedder)
        assert result["orphans_removed"] >= 1, "should remove orphaned vectors"
        assert store.count() == 0, "orphaned vectors should be gone"

    def test_enqueues_stale_reindex(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = create_item(conn, "old model note")
        index_item(conn, store, old_embedder, item.id)

        result = on_startup(conn, store, fake_embedder)
        assert result["stale_reindex_enqueued"] == 1, "should enqueue reindex for stale item"

    def test_recovers_stalled_jobs(self, conn, store, fake_embedder):
        enqueue(conn, "embed", {"item_id": "some-id"})
        conn.execute(
            "UPDATE jobs SET status = 'processing' WHERE status = 'pending'"
        )
        conn.commit()

        result = on_startup(conn, store, fake_embedder)
        assert result["stalled_recovered"] >= 0, "should report stalled job recovery"

        status = queue_status(conn)
        assert status.get("processing", 0) == 0, "no jobs should be stuck in processing"

    def test_returns_summary(self, conn, store, fake_embedder):
        result = on_startup(conn, store, fake_embedder)
        assert "stalled_recovered" in result, "summary should include stalled_recovered"
        assert "orphans_removed" in result, "summary should include orphans_removed"
        assert "stale_reindex_enqueued" in result, "summary should include stale_reindex_enqueued"

    def test_idempotent(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = create_item(conn, "note")
        index_item(conn, store, old_embedder, item.id)

        first = on_startup(conn, store, fake_embedder)
        process_all_pending(conn, store, fake_embedder)
        second = on_startup(conn, store, fake_embedder)

        assert second["orphans_removed"] == 0, "second startup should find no orphans"
        assert second["stale_reindex_enqueued"] == 0, "second startup should find nothing stale"
