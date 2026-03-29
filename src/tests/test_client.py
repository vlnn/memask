import json
import threading

import pytest

from memask.app import AppContext
from memask.client import DaemonClient
from memask.server import create_app
from tests.helpers import FakeEmbeddingService, FakeLLM, FakeReranker


@pytest.fixture
def live_server(tmp_path):
    ctx = AppContext(
        db_path=str(tmp_path / "test.db"),
        embedder_factory=lambda: FakeEmbeddingService(),
        llm_factory=lambda: FakeLLM(),
        reranker_factory=lambda: FakeReranker(),
    )
    app = create_app(ctx)
    app.config["TESTING"] = True

    server = _start_test_server(app)
    yield server
    ctx.shutdown()


def _start_test_server(app):
    from werkzeug.serving import make_server

    srv = make_server("127.0.0.1", 0, app)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return {"port": port, "server": srv}


class TestDaemonClientHealth:
    def test_is_running(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        assert client.is_running() is True, "should detect running daemon"

    def test_not_running(self):
        client = DaemonClient("http://127.0.0.1:1")
        assert client.is_running() is False, "should detect missing daemon"

    def test_health_returns_status(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        data = client.health()
        assert data["status"] == "ok", "health should report ok"


class TestDaemonClientInput:
    def test_capture(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        result = client.input("meeting went well")
        assert result["action"] == "captured", "should capture note"

    def test_todo_create(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        result = client.input("remind me to buy milk")
        assert result["action"] == "todo_created", "should create todo"

    def test_search(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        client.input("deployment notes here")
        result = client.input("?deployment")
        assert result["action"] in ("searched", "answered"), (
            "should search or answer"
        )


class TestDaemonClientItems:
    def test_lists_items(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        client.input("a note")
        result = client.items()
        assert len(result["items"]) >= 1, "should list items"

    def test_filters_by_type(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        client.input("remind me to buy milk")
        client.input("plain note")
        result = client.items(type="todo")
        for item in result["items"]:
            assert item["type"] == "todo", "should only return todos"


class TestDaemonClientSearch:
    def test_finds_results(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        client.input("deployment pipeline ready")
        result = client.search("deployment")
        assert len(result["results"]) > 0, "should find matching items"

    def test_empty_results(self, live_server):
        client = DaemonClient(f"http://127.0.0.1:{live_server['port']}")
        result = client.search("nonexistent")
        assert result["results"] == [], "should return empty list"


class TestDaemonClientConnectionError:
    def test_raises_on_unreachable(self):
        client = DaemonClient("http://127.0.0.1:1")
        with pytest.raises(ConnectionError):
            client.health()
