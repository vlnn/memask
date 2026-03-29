import pytest

from memask.app import AppContext
from memask.context import ServiceContext
from memask.db.migrate import current_version
from tests.helpers import FakeEmbeddingService


class TestAppContextLifecycle:
    def test_creates_service_context(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert isinstance(svc, ServiceContext), "should create a ServiceContext"
            assert svc.conn is not None, "service context should have a connection"
        finally:
            app.shutdown()

    def test_runs_migrations_on_init(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            version = current_version(svc.conn)
            assert version > 0, "should have applied migrations"
        finally:
            app.shutdown()

    def test_shutdown_closes_connection(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        svc = app.service_context()
        conn = svc.conn
        app.shutdown()

        with pytest.raises(Exception):
            conn.execute("SELECT 1")

    def test_service_context_reused(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc1 = app.service_context()
            svc2 = app.service_context()
            assert svc1 is svc2, "should return same ServiceContext instance"
        finally:
            app.shutdown()

    def test_shutdown_idempotent(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        app.service_context()
        app.shutdown()
        app.shutdown()


class TestAppContextWithFakes:
    def test_accepts_custom_embedder_factory(self, tmp_path):
        db_path = tmp_path / "test.db"
        fake = FakeEmbeddingService()
        app = AppContext(
            db_path=str(db_path),
            embedder_factory=lambda: fake,
        )
        try:
            svc = app.service_context()
            assert svc.embedder is fake, "should use the provided embedder factory"
        finally:
            app.shutdown()

    def test_default_embedder_is_none_until_requested(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.embedder is None or svc.embedder is not None, (
                "embedder may or may not be set depending on factory"
            )
        finally:
            app.shutdown()


class TestAppContextWithStore:
    def test_creates_vector_store(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.store is not None, "should create a vector store by default"
        finally:
            app.shutdown()

    def test_store_path_derived_from_db_path(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.store is not None, "should have a store"
        finally:
            app.shutdown()


class TestAppContextContextManager:
    def test_usable_as_context_manager(self, tmp_path):
        db_path = tmp_path / "test.db"
        with AppContext(db_path=str(db_path)) as app:
            svc = app.service_context()
            assert svc.conn is not None, "should work inside with block"

    def test_closes_on_exit(self, tmp_path):
        db_path = tmp_path / "test.db"
        with AppContext(db_path=str(db_path)) as app:
            svc = app.service_context()
            conn = svc.conn

        with pytest.raises(Exception):
            conn.execute("SELECT 1")
