from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from memask.context import ServiceContext
from memask.db.connection import get_connection
from memask.db.migrate import migrate_up
from memask.rag.llm import LocalLLM
from memask.rag.models import model_path
from memask.rag.session import Session
from memask.search.vector_store import VectorStore


class AppContext:
    def __init__(
        self,
        db_path: str | None = None,
        embedder_factory: Callable[[], Any] | None = None,
        llm_factory: Callable[[], Any] | None = None,
        reranker_factory: Callable[[], Any] | None = None,
        lance_path: str | Path | None = None,
    ):
        self._db_path = db_path
        self._embedder_factory = embedder_factory
        self._llm_factory = llm_factory
        self._reranker_factory = reranker_factory
        self._lance_path = lance_path
        self._svc: ServiceContext | None = None
        self._shutdown = False

    def service_context(self) -> ServiceContext:
        if self._svc is not None:
            return self._svc

        conn = get_connection(self._db_path)
        migrate_up(conn)

        store = VectorStore(self._resolve_lance_path())

        embedder = None
        if self._embedder_factory is not None:
            embedder = self._embedder_factory()

        llm = self._build_llm()
        reranker = self._build_reranker()

        self._svc = ServiceContext(
            conn=conn,
            store=store,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            session=Session(),
        )
        return self._svc

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._svc is not None and self._svc.conn is not None:
            try:
                self._svc.conn.close()
            except Exception:
                pass

    def _build_llm(self):
        if self._llm_factory is not None:
            return self._llm_factory()
        path = model_path(base=self._resolve_memask_dir())
        return LocalLLM(model_path=str(path))

    def _build_reranker(self):
        if self._reranker_factory is not None:
            return self._reranker_factory()
        return None

    def _resolve_lance_path(self) -> Path:
        if self._lance_path:
            return Path(self._lance_path)
        if self._db_path:
            return Path(self._db_path).parent / "vectors"
        return Path.home() / ".memask" / "vectors"

    def _resolve_memask_dir(self) -> Path:
        if self._db_path:
            return Path(self._db_path).parent
        return Path.home() / ".memask"

    def __enter__(self) -> AppContext:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
