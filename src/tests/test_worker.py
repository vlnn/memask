import json

import pytest

from memask.repository.items import create_item, soft_delete_item
from memask.repository.jobs import dequeue, enqueue, get_job, queue_status
from memask.search.vector_store import VectorStore
from memask.search.worker import process_next_job, enqueue_embedding, process_all_pending


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


class TestEnqueueEmbedding:
    def test_creates_embed_job(self, conn):
        item = create_item(conn, "hello world")
        job = enqueue_embedding(conn, item.id)
        assert job.type == "embed", "should create an embed job"
        assert job.status == "pending", "job should start as pending"
        payload = json.loads(job.payload)
        assert payload["item_id"] == item.id, "payload should reference the item"

    def test_creates_delete_vectors_job(self, conn):
        from memask.search.worker import enqueue_delete_vectors
        job = enqueue_delete_vectors(conn, "item-123")
        assert job.type == "delete_vectors", "should create a delete_vectors job"
        payload = json.loads(job.payload)
        assert payload["item_id"] == "item-123", "payload should reference the item"


class TestProcessNextJob:
    def test_processes_embed_job(self, conn, store, fake_embedder):
        item = create_item(conn, "deploy to production")
        enqueue_embedding(conn, item.id)

        processed = process_next_job(conn, store, fake_embedder)
        assert processed is True, "should process one job"
        assert store.count() >= 1, "item should be indexed in vector store"

    def test_marks_job_complete(self, conn, store, fake_embedder):
        item = create_item(conn, "test note")
        job = enqueue_embedding(conn, item.id)

        process_next_job(conn, store, fake_embedder)
        updated = get_job(conn, job.id)
        assert updated.status == "complete", "job should be marked complete after success"

    def test_returns_false_when_queue_empty(self, conn, store, fake_embedder):
        processed = process_next_job(conn, store, fake_embedder)
        assert processed is False, "should return False when no pending jobs"

    def test_handles_missing_item_gracefully(self, conn, store, fake_embedder):
        enqueue(conn, "embed", {"item_id": "nonexistent-id"})
        processed = process_next_job(conn, store, fake_embedder)
        assert processed is True, "should process the job even if item is gone"
        status = queue_status(conn)
        assert status.get("complete", 0) == 1, "job should be complete (item gone is not an error)"

    def test_retries_on_failure(self, conn, store):
        from conftest import FakeEmbeddingService

        class FailingEmbedder(FakeEmbeddingService):
            def embed_one(self, text):
                raise RuntimeError("model crashed")

        item = create_item(conn, "will fail")
        job = enqueue_embedding(conn, item.id)

        process_next_job(conn, store, FailingEmbedder())
        updated = get_job(conn, job.id)
        assert updated.status == "pending", "failed job under max attempts should be requeued"
        assert updated.last_error is not None, "should record the error"

    def test_permanently_fails_after_max_attempts(self, conn, store):
        from conftest import FakeEmbeddingService

        class FailingEmbedder(FakeEmbeddingService):
            def embed_one(self, text):
                raise RuntimeError("model crashed")

        item = create_item(conn, "will fail permanently")
        enqueue_embedding(conn, item.id, max_attempts=1)

        process_next_job(conn, store, FailingEmbedder())
        process_next_job(conn, store, FailingEmbedder())

        status = queue_status(conn)
        assert status.get("failed", 0) == 1, "should be permanently failed after max attempts"

    def test_processes_delete_vectors_job(self, conn, store, fake_embedder):
        from memask.search.worker import enqueue_delete_vectors

        item = create_item(conn, "will be removed from index")
        enqueue_embedding(conn, item.id)
        process_next_job(conn, store, fake_embedder)
        assert store.count() >= 1, "item should be indexed first"

        enqueue_delete_vectors(conn, item.id)
        process_next_job(conn, store, fake_embedder)
        assert item.id not in store.list_item_ids(), "vectors should be removed"


class TestProcessAllPending:
    def test_processes_multiple_jobs(self, conn, store, fake_embedder):
        for i in range(5):
            item = create_item(conn, f"note number {i}")
            enqueue_embedding(conn, item.id)

        count = process_all_pending(conn, store, fake_embedder)
        assert count == 5, "should process all 5 pending jobs"
        assert store.count() >= 5, "all items should be indexed"

    def test_returns_zero_when_empty(self, conn, store, fake_embedder):
        count = process_all_pending(conn, store, fake_embedder)
        assert count == 0, "should return 0 when no jobs to process"

    def test_stops_at_max_batch(self, conn, store, fake_embedder):
        for i in range(10):
            item = create_item(conn, f"note {i}")
            enqueue_embedding(conn, item.id)

        count = process_all_pending(conn, store, fake_embedder, max_batch=3)
        assert count == 3, "should stop at max_batch"

        remaining = queue_status(conn)
        assert remaining.get("pending", 0) == 7, "remaining jobs should still be pending"
