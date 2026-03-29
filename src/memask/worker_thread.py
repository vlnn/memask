from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 5.0


def start_background_worker(
    db_path: str | None,
    store: Any,
    embedder: Any,
    interval: float = DEFAULT_INTERVAL,
) -> threading.Thread:
    thread = threading.Thread(
        target=_worker_loop,
        args=(db_path, store, embedder, interval),
        daemon=True,
        name="memask-embedding-worker",
    )
    thread.start()
    logger.info("background worker started (interval=%.1fs)", interval)
    return thread


def _worker_loop(
    db_path: str | None,
    store: Any,
    embedder: Any,
    interval: float,
) -> None:
    from memask.db.connection import get_connection
    from memask.db.migrate import migrate_up
    from memask.search.worker import process_all_pending

    conn = get_connection(db_path)
    migrate_up(conn)

    while True:
        try:
            if store and embedder:
                processed = process_all_pending(conn, store, embedder)
                if processed > 0:
                    logger.info("background worker processed %d jobs", processed)
        except Exception:
            logger.exception("background worker error")
        time.sleep(interval)
