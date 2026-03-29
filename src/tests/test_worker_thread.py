import time

import pytest

from memask.app import AppContext
from memask.repository.items import create_item
from memask.repository.jobs import queue_status
from memask.search.worker import enqueue_embedding
from memask.worker_thread import start_background_worker
from tests.helpers import FakeEmbeddingService


class TestBackgroundWorker:
    def test_processes_pending_jobs(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        embedder = FakeEmbeddingService()
        ctx = AppContext(
            db_path=db_path,
            embedder_factory=lambda: embedder,
        )
        try:
            svc = ctx.service_context()
            item = create_item(svc.conn, "background test note")
            enqueue_embedding(svc.conn, item.id)

            start_background_worker(db_path, svc.store, embedder, interval=0.1)
            time.sleep(0.5)

            status = queue_status(svc.conn)
            assert status.get("pending", 0) == 0, (
                "worker should process pending jobs"
            )
            assert status.get("complete", 0) == 1, (
                "job should be marked complete"
            )
        finally:
            ctx.shutdown()

    def test_thread_is_daemon(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ctx = AppContext(
            db_path=db_path,
            embedder_factory=lambda: FakeEmbeddingService(),
        )
        try:
            svc = ctx.service_context()
            thread = start_background_worker(db_path, svc.store, svc.embedder, interval=60)
            assert thread.daemon is True, (
                "worker thread should be a daemon thread"
            )
        finally:
            ctx.shutdown()

    def test_survives_without_embedder(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        ctx = AppContext(db_path=db_path)
        try:
            ctx.service_context()
            thread = start_background_worker(db_path, None, None, interval=0.1)
            time.sleep(0.3)
            assert thread.is_alive(), (
                "worker should keep running even without embedder"
            )
        finally:
            ctx.shutdown()
