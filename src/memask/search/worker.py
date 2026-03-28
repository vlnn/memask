import json
import logging
import sqlite3

from memask.models.job import Job
from memask.repository.jobs import (
    dequeue,
    enqueue,
    mark_complete,
    requeue_if_retriable,
)
from memask.search.embedding import EmbeddingService
from memask.search.indexer import index_item
from memask.search.vector_store import VectorStore

logger = logging.getLogger(__name__)

WORKER_JOB_TYPES = {"embed", "delete_vectors"}


def enqueue_embedding(
    conn: sqlite3.Connection,
    item_id: str,
    *,
    max_attempts: int = 3,
) -> Job:
    return enqueue(conn, "embed", {"item_id": item_id}, max_attempts=max_attempts)


def enqueue_delete_vectors(
    conn: sqlite3.Connection,
    item_id: str,
) -> Job:
    return enqueue(conn, "delete_vectors", {"item_id": item_id})


def process_next_job(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
) -> bool:
    job = _dequeue_worker_job(conn)
    if job is None:
        return False

    try:
        _dispatch(conn, vector_store, embedding_service, job)
        mark_complete(conn, job.id)
        logger.debug("job %s (%s) completed", job.id, job.type)
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning("job %s (%s) failed: %s", job.id, job.type, error_msg)
        requeue_if_retriable(conn, job.id, error_msg)

    return True


def process_all_pending(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    *,
    max_batch: int = 100,
) -> int:
    count = 0
    while count < max_batch:
        if not process_next_job(conn, vector_store, embedding_service):
            break
        count += 1
    return count


def _dequeue_worker_job(conn: sqlite3.Connection) -> Job | None:
    for job_type in WORKER_JOB_TYPES:
        job = dequeue(conn, job_type)
        if job is not None:
            return job
    return None


def _dispatch(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    job: Job,
) -> None:
    payload = json.loads(job.payload) if job.payload else {}
    item_id = payload.get("item_id")

    if job.type == "embed":
        _handle_embed(conn, vector_store, embedding_service, item_id)
    elif job.type == "delete_vectors":
        _handle_delete_vectors(vector_store, item_id)
    else:
        raise ValueError(f"unknown job type: {job.type}")


def _handle_embed(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    item_id: str | None,
) -> None:
    if not item_id:
        logger.warning("embed job missing item_id")
        return
    indexed = index_item(conn, vector_store, embedding_service, item_id)
    logger.debug("indexed item %s: %d vectors", item_id, indexed)


def _handle_delete_vectors(
    vector_store: VectorStore,
    item_id: str | None,
) -> None:
    if not item_id:
        logger.warning("delete_vectors job missing item_id")
        return
    vector_store.delete_by_item(item_id)
    logger.debug("deleted vectors for item %s", item_id)
