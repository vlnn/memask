from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

EMBEDDINGS_TABLE = "embeddings"


class VectorStore:
    def __init__(self, db_path: str | Path, dimension: int = 384):
        self._db = lancedb.connect(str(db_path))
        self._dimension = dimension
        self._schema = pa.schema([
            pa.field("item_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("model_version", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dimension)),
        ])
        self._ensure_table()

    def _ensure_table(self):
        if EMBEDDINGS_TABLE not in self._db.list_tables():
            self._db.create_table(EMBEDDINGS_TABLE, schema=self._schema)

    def _table(self):
        return self._db.open_table(EMBEDDINGS_TABLE)

    def add(
        self,
        item_id: str,
        chunk_index: int,
        text: str,
        vector: np.ndarray,
        model_version: str,
    ) -> None:
        table = self._table()
        table.add([{
            "item_id": item_id,
            "chunk_index": chunk_index,
            "text": text,
            "model_version": model_version,
            "vector": vector.tolist(),
        }])

    def add_batch(self, records: list[dict]) -> None:
        if not records:
            return
        table = self._table()
        table.add(records)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        limit: int = 20,
        model_version: str | None = None,
    ) -> list[dict]:
        table = self._table()
        query = table.search(query_vector.tolist(), vector_column_name="vector").limit(limit)

        if model_version:
            query = query.where(f"model_version = '{model_version}'")

        results = query.to_list()
        return [
            {
                "item_id": r["item_id"],
                "chunk_index": r["chunk_index"],
                "text": r["text"],
                "model_version": r["model_version"],
                "distance": r.get("_distance", 0.0),
            }
            for r in results
        ]

    def delete_by_item(self, item_id: str) -> None:
        table = self._table()
        table.delete(f"item_id = '{item_id}'")

    def delete_by_model(self, model_version: str) -> None:
        table = self._table()
        table.delete(f"model_version = '{model_version}'")

    def list_item_ids(self) -> set[str]:
        table = self._table()
        arrow_table = table.to_arrow()
        if arrow_table.num_rows == 0:
            return set()
        return set(arrow_table.column("item_id").to_pylist())

    def count(self) -> int:
        table = self._table()
        return table.count_rows()

    def stale_items(self, current_model: str) -> set[str]:
        table = self._table()
        arrow_table = table.to_arrow()
        if arrow_table.num_rows == 0:
            return set()
        models = arrow_table.column("model_version").to_pylist()
        item_ids = arrow_table.column("item_id").to_pylist()
        return {
            item_id
            for item_id, model in zip(item_ids, models)
            if model != current_model
        }
