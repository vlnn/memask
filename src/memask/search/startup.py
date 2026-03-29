import json
import sqlite3

from memask.repository.jobs import recover_stalled
from memask.search.indexer import reconcile
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding


def on_startup(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service,
) -> dict[str, int]:
    stalled = recover_stalled(conn)
    orphans = reconcile(conn, vector_store)
    stale = enqueue_stale_reindex(
        conn,
        vector_store,
        embedding_service,
    )

    return {
        "stalled_recovered": stalled,
        "orphans_removed": orphans,
        "stale_reindex_enqueued": stale,
    }


def enqueue_stale_reindex(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service,
) -> int:
    stale = vector_store.stale_items(embedding_service.model_name)
    if not stale:
        return 0

    rows = conn.execute(
        "SELECT payload FROM jobs WHERE type = 'embed' AND status = 'pending'",
    ).fetchall()
    already_queued = set()
    for r in rows:
        if r["payload"]:
            payload = json.loads(r["payload"])
            if "item_id" in payload:
                already_queued.add(payload["item_id"])

    count = 0
    for item_id in stale:
        if item_id not in already_queued:
            enqueue_embedding(conn, item_id)
            count += 1
    return count
