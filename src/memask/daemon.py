from __future__ import annotations

import logging

from memask.app import AppContext
from memask.search.embedding import EmbeddingService
from memask.search.startup import on_startup
from memask.server import create_app
from memask.worker_thread import start_background_worker

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7394
DEFAULT_HOST = "127.0.0.1"


def run_daemon(
    *,
    db_path: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app_context = AppContext(
        db_path=db_path,
        embedder_factory=lambda: EmbeddingService(),
    )

    svc = app_context.service_context()
    logger.info("running startup checks")
    if svc.store and svc.embedder:
        result = on_startup(svc.conn, svc.store, svc.embedder)
        logger.info(
            "startup: recovered=%d orphans=%d stale=%d",
            result["stalled_recovered"],
            result["orphans_removed"],
            result["stale_reindex_enqueued"],
        )

    start_background_worker(db_path, svc.store, svc.embedder)

    app = create_app(app_context)
    logger.info("starting daemon on %s:%d", host, port)
    try:
        app.run(host=host, port=port)
    finally:
        app_context.shutdown()
