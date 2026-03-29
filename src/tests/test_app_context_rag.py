import pytest

from memask.app import AppContext
from memask.rag.session import Session
from tests.helpers import FakeEmbeddingService, FakeLLM, FakeReranker


class TestAppContextLLMWiring:
    def test_creates_llm_by_default(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.llm is not None, "should create an LLM by default"
        finally:
            app.shutdown()

    def test_default_llm_unavailable_without_model_file(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.llm.is_available() is False, (
                "default LLM should be unavailable when model file missing"
            )
        finally:
            app.shutdown()

    def test_accepts_custom_llm_factory(self, tmp_path):
        db_path = tmp_path / "test.db"
        fake = FakeLLM(response="hello")
        app = AppContext(
            db_path=str(db_path),
            llm_factory=lambda: fake,
        )
        try:
            svc = app.service_context()
            assert svc.llm is fake, "should use the provided LLM factory"
            assert svc.llm.is_available() is True, "fake LLM should be available"
        finally:
            app.shutdown()


class TestAppContextSessionWiring:
    def test_creates_session(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.session is not None, "should create a session"
            assert isinstance(svc.session, Session), "session should be a Session"
        finally:
            app.shutdown()

    def test_session_starts_empty(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert len(svc.session) == 0, "session should start empty"
        finally:
            app.shutdown()

    def test_session_resets_on_new_app(self, tmp_path):
        db_path = tmp_path / "test.db"

        app1 = AppContext(db_path=str(db_path))
        svc1 = app1.service_context()
        svc1.session.add_exchange("q", "a")
        app1.shutdown()

        app2 = AppContext(db_path=str(db_path))
        try:
            svc2 = app2.service_context()
            assert len(svc2.session) == 0, (
                "new AppContext should have fresh session"
            )
        finally:
            app2.shutdown()


class TestAppContextRerankerWiring:
    def test_no_reranker_by_default(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(db_path=str(db_path))
        try:
            svc = app.service_context()
            assert svc.reranker is None, (
                "should not create reranker by default"
            )
        finally:
            app.shutdown()

    def test_accepts_custom_reranker_factory(self, tmp_path):
        db_path = tmp_path / "test.db"
        fake = FakeReranker()
        app = AppContext(
            db_path=str(db_path),
            reranker_factory=lambda: fake,
        )
        try:
            svc = app.service_context()
            assert svc.reranker is fake, (
                "should use the provided reranker factory"
            )
        finally:
            app.shutdown()


class TestAppContextFullyWired:
    def test_all_components_present(self, tmp_path):
        db_path = tmp_path / "test.db"
        app = AppContext(
            db_path=str(db_path),
            embedder_factory=lambda: FakeEmbeddingService(),
            llm_factory=lambda: FakeLLM(),
            reranker_factory=lambda: FakeReranker(),
        )
        try:
            svc = app.service_context()
            assert svc.conn is not None, "should have conn"
            assert svc.store is not None, "should have store"
            assert svc.embedder is not None, "should have embedder"
            assert svc.llm is not None, "should have llm"
            assert svc.reranker is not None, "should have reranker"
            assert svc.session is not None, "should have session"
        finally:
            app.shutdown()
