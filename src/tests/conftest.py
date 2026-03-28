import pytest

import sqlite3
import numpy as np

from memask.db.migrate import migrate_up


@pytest.fixture
def raw_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn
    conn.close()


@pytest.fixture
def conn(raw_conn):
    migrate_up(raw_conn)
    return raw_conn


@pytest.fixture
def lance_dir(tmp_path):
    return tmp_path / "lance"


class FakeEmbeddingService:
    def __init__(self, dimension=384, model_name="test-model"):
        self._model_name = model_name
        self._dimension = dimension
        self._call_log = []

    @property
    def model_name(self):
        return self._model_name

    @property
    def dimension(self):
        return self._dimension

    @property
    def call_log(self):
        return self._call_log

    def embed_one(self, text):
        self._call_log.append(("embed_one", text))
        return self._deterministic_vector(text)

    def embed_many(self, texts):
        self._call_log.append(("embed_many", texts))
        return np.array([self._deterministic_vector(t) for t in texts])

    def _deterministic_vector(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingService()
