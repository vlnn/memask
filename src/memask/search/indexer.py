import sqlite3

from memask.search.chunking import chunk_text, needs_chunking
from memask.search.embedding import EmbeddingService
from memask.search.vector_store import VectorStore


def index_item(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    item_id: str,
) -> int:
    row = conn.execute(
        "SELECT id, content, title FROM items WHERE id = ? AND deleted_at IS NULL",
        (item_id,),
    ).fetchone()

    if not row:
        return 0

    text = _item_text(row["content"], row["title"])

    vector_store.delete_by_item(item_id)

    if needs_chunking(text):
        return _index_chunked(vector_store, embedding_service, item_id, text)
    return _index_whole(vector_store, embedding_service, item_id, text)


def _index_whole(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    item_id: str,
    text: str,
) -> int:
    vector = embedding_service.embed_one(text)
    vector_store.add(
        item_id=item_id,
        chunk_index=0,
        text=text,
        vector=vector,
        model_version=embedding_service.model_name,
    )
    return 1


def _index_chunked(
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
    item_id: str,
    text: str,
) -> int:
    chunks = chunk_text(item_id, text)
    texts = [c.text for c in chunks]
    vectors = embedding_service.embed_many(texts)

    records = [
        {
            "item_id": chunk.item_id,
            "chunk_index": chunk.index,
            "text": chunk.text,
            "model_version": embedding_service.model_name,
            "vector": vectors[i].tolist(),
        }
        for i, chunk in enumerate(chunks)
    ]

    vector_store.add_batch(records)
    return len(chunks)


def reconcile(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
) -> int:
    active_ids = _active_item_ids(conn)
    indexed_ids = vector_store.list_item_ids()
    orphaned = indexed_ids - active_ids

    for item_id in orphaned:
        vector_store.delete_by_item(item_id)

    return len(orphaned)


def reindex_stale(
    conn: sqlite3.Connection,
    vector_store: VectorStore,
    embedding_service: EmbeddingService,
) -> int:
    stale = vector_store.stale_items(embedding_service.model_name)
    count = 0
    for item_id in stale:
        count += index_item(conn, vector_store, embedding_service, item_id)
    return count


def _active_item_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT id FROM items WHERE deleted_at IS NULL").fetchall()
    return {row["id"] for row in rows}


def _item_text(content: str, title: str | None) -> str:
    if title:
        return f"{title}\n{content}"
    return content
