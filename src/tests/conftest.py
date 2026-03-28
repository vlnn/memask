import pytest

from memask.db.connection import get_connection
from memask.db.migrate import migrate_up


@pytest.fixture
def conn():
    connection = get_connection(":memory:")
    migrate_up(connection)
    yield connection
    connection.close()


@pytest.fixture
def raw_conn():
    connection = get_connection(":memory:")
    yield connection
    connection.close()
