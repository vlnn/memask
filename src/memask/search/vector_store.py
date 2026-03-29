import logging
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)

TABLE_NAME = "embeddings"


class VectorStore:
    def __init__(self, path: Path | str, dimension: int = 384):
        self._path = Path(path)
        self._dimension = dimension
        self._db = None
        self._table = None

    def _get_db(self):
        if self._db is None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self._path))
        return self._db

    def _get_table(self):
        if self._table is not None:
            return self._table

        db = self._get_db()

        try:
            self._table = db.open_table(TABLE_NAME)
            return self._table
        except Exception:
            pass

        schema = pa.schema(
            [
                pa.field("item_id", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("model_version", pa.string()),
                pa.field(
                    "vector",
                    pa.list_(pa.float32(), self._dimension),
                ),
            ]
        )
        self._table = db.create_table(
            TABLE_NAME,
            schema=schema,
        )
        return self._table

    def add(
        self,
        item_id: str,
        chunk_index: int,
        text: str,
        vector: np.ndarray,
        model_version: str,
    ) -> None:
        table = self._get_table()
        table.add(
            [
                {
                    "item_id": item_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "model_version": model_version,
                    "vector": vector.tolist() if hasattr(vector, "tolist") else vector,
                }
            ]
        )

    def add_batch(self, records: list[dict]) -> None:
        if not records:
            return
        table = self._get_table()
        table.add(records)

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 20,
        model_version: str | None = None,
    ) -> list[dict]:
        table = self._get_table()
        query = table.search(query_vector.tolist()).limit(limit)
        if model_version:
            query = query.where(
                f"model_version = '{model_version}'",
            )
        results = query.to_arrow()
        rows = []
        for i in range(len(results)):
            rows.append(
                {
                    "item_id": results.column("item_id")[i].as_py(),
                    "chunk_index": results.column("chunk_index")[i].as_py(),
                    "text": results.column("text")[i].as_py(),
                    "score": results.column("_distance")[i].as_py(),
                }
            )
        return rows

    def delete_by_item(self, item_id: str) -> None:
        table = self._get_table()
        table.delete(f"item_id = '{item_id}'")

    def delete_by_model(self, model_version: str) -> None:
        table = self._get_table()
        table.delete(f"model_version = '{model_version}'")

    def list_item_ids(self) -> set[str]:
        table = self._get_table()
        result = table.to_arrow().column("item_id")
        return {v.as_py() for v in result}

    def count(self) -> int:
        table = self._get_table()
        return table.count_rows()

    def stale_items(self, current_model: str) -> set[str]:
        table = self._get_table()
        arrow = table.to_arrow()
        if len(arrow) == 0:
            return set()
        models = arrow.column("model_version")
        item_ids = arrow.column("item_id")
        return {
            item_ids[i].as_py()
            for i in range(len(arrow))
            if models[i].as_py() != current_model
        }
