import sqlite3

import pytest

from memask.db.migrate import migrate_up
from tests.helpers import FakeEmbeddingService


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


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingService()
