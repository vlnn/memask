import json
import logging
import sqlite3

from memask.repository.jobs import recover_stalled
from memask.search.embedding import EmbeddingService
from memask.search.indexer import reconcile
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding

logger = logging.getLogger(__name__)


def on_startup(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
) -> dict[str, int]:
    stalled = recover_stalled(conn)
    logger.info("recovered %d stalled jobs", stalled)

    orphans = reconcile(conn, vector_store)
    logger.info("removed %d orphaned vectors", orphans)

    reindex_count = enqueue_stale_reindex(conn, vector_store, embedding_service)
    logger.info("enqueued %d items for reindexing", reindex_count)

    return {
        "stalled_recovered": stalled,
        "orphans_removed": orphans,
        "stale_reindex_enqueued": reindex_count,
    }


def enqueue_stale_reindex(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
) -> int:
    stale_ids = vector_store.stale_items(embedding_service.model_name)
    if not stale_ids:
        return 0

    already_queued = _pending_embed_item_ids(conn)
    to_enqueue = stale_ids - already_queued

    for item_id in to_enqueue:
        enqueue_embedding(conn, item_id)

    return len(to_enqueue)


def _pending_embed_item_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT payload FROM jobs WHERE type = 'embed' AND status = 'pending'"
    ).fetchall()
    ids = set()
    for row in rows:
        if row["payload"]:
            payload = json.loads(row["payload"])
            item_id = payload.get("item_id")
            if item_id:
                ids.add(item_id)
    return ids
