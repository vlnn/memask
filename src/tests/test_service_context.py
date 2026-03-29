from memask.context import ServiceContext
from memask.search.vector_store import VectorStore


class TestServiceContextConstruction:
    def test_requires_conn(self, conn):
        svc = ServiceContext(conn=conn)
        assert svc.conn is conn, "should hold the connection"

    def test_optional_store(self, conn, lance_dir):
        store = VectorStore(lance_dir)
        svc = ServiceContext(conn=conn, store=store)
        assert svc.store is store, "should hold the vector store"

    def test_optional_embedder(self, conn, fake_embedder):
        svc = ServiceContext(conn=conn, embedder=fake_embedder)
        assert svc.embedder is fake_embedder, "should hold the embedder"

    def test_store_defaults_to_none(self, conn):
        svc = ServiceContext(conn=conn)
        assert svc.store is None, "store should default to None"

    def test_embedder_defaults_to_none(self, conn):
        svc = ServiceContext(conn=conn)
        assert svc.embedder is None, "embedder should default to None"

    def test_llm_defaults_to_none(self, conn):
        svc = ServiceContext(conn=conn)
        assert svc.llm is None, "llm should default to None"

    def test_session_defaults_to_none(self, conn):
        svc = ServiceContext(conn=conn)
        assert svc.session is None, "session should default to None"


class TestServiceContextFullyWired:
    def test_all_components(self, conn, lance_dir, fake_embedder):
        store = VectorStore(lance_dir)
        svc = ServiceContext(
            conn=conn,
            store=store,
            embedder=fake_embedder,
        )
        assert svc.conn is conn, "should hold conn"
        assert svc.store is store, "should hold store"
        assert svc.embedder is fake_embedder, "should hold embedder"
