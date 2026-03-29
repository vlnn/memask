import json

import pytest

from memask.app import AppContext
from memask.server import create_app
from memask.context import ServiceContext
from memask.repository.items import create_item
from memask.router.dispatcher import dispatch
from tests.helpers import FakeEmbeddingService, FakeLLM, FakeReranker


@pytest.fixture
def app_context(tmp_path):
    ctx = AppContext(
        db_path=str(tmp_path / "test.db"),
        embedder_factory=lambda: FakeEmbeddingService(),
        llm_factory=lambda: FakeLLM(),
        reranker_factory=lambda: FakeReranker(),
    )
    yield ctx
    ctx.shutdown()


@pytest.fixture
def client(app_context):
    app = create_app(app_context)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def resp_json(resp):
    return json.loads(resp.data)


class TestErrorHandling:
    def test_unhandled_error_returns_json(self, app_context, mocker):
        mocker.patch(
            "memask.router.dispatcher.dispatch",
            side_effect=RuntimeError("kaboom"),
        )
        app = create_app(app_context)
        app.config["TESTING"] = False
        with app.test_client() as c:
            resp = c.post("/input", json={"text": "hello"})
            assert resp.status_code == 500, "unhandled error should return 500"
            data = resp_json(resp)
            assert data["action"] == "error", "should return error action"
            assert "kaboom" in data["data"]["message"], (
                "should include error message"
            )

    def test_error_response_has_cors_headers(self, app_context, mocker):
        mocker.patch(
            "memask.router.dispatcher.dispatch",
            side_effect=RuntimeError("boom"),
        )
        app = create_app(app_context)
        app.config["TESTING"] = False
        with app.test_client() as c:
            resp = c.post("/input", json={"text": "hello"})
            assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
                "error responses should include CORS headers"
            )

    def test_error_response_is_json_content_type(self, app_context, mocker):
        mocker.patch(
            "memask.router.dispatcher.dispatch",
            side_effect=ValueError("bad"),
        )
        app = create_app(app_context)
        app.config["TESTING"] = False
        with app.test_client() as c:
            resp = c.post("/input", json={"text": "hello"})
            assert resp.content_type == "application/json", (
                "error should return JSON content type"
            )


class TestListCommand:
    def test_list_returns_recent_items(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "first note")
        create_item(conn, "second note")
        result = dispatch(svc, "/list")
        assert result.action == "listed", "should return listed action"
        assert len(result.data["items"]) == 2, "should list all items"

    def test_list_returns_newest_first(self, conn):
        svc = ServiceContext(conn=conn)
        create_item(conn, "older")
        create_item(conn, "newer")
        result = dispatch(svc, "/list")
        assert result.data["items"][0]["content"] == "newer", (
            "newest item should be first"
        )

    @pytest.mark.parametrize("command,expected_type", [
        ("/list notes", "note"),
        ("/list todos", "todo"),
    ])
    def test_list_filters_by_type(self, conn, command, expected_type):
        svc = ServiceContext(conn=conn)
        create_item(conn, "a note", type="note")
        create_item(conn, "a todo", type="todo", status="pending")
        result = dispatch(svc, command)
        assert result.action == "listed", f"'{command}' should return listed action"
        for item in result.data["items"]:
            assert item["type"] == expected_type, (
                f"'{command}' should only return {expected_type} items"
            )

    def test_list_empty(self, conn):
        svc = ServiceContext(conn=conn)
        result = dispatch(svc, "/list")
        assert result.action == "listed", "should return listed even when empty"
        assert result.data["items"] == [], "should return empty list"

    def test_list_via_http(self, client):
        client.post("/input", json={"text": "a note"})
        client.post("/input", json={"text": "remind me to buy milk"})
        resp = client.post("/input", json={"text": "/list"})
        data = resp_json(resp)
        assert data["action"] == "listed", "should return listed via HTTP"
        assert len(data["data"]["items"]) >= 2, "should list items via HTTP"

    def test_list_notes_via_http(self, client):
        client.post("/input", json={"text": "just a note"})
        client.post("/input", json={"text": "remind me to buy milk"})
        resp = client.post("/input", json={"text": "/list notes"})
        data = resp_json(resp)
        assert data["action"] == "listed", "should list notes via HTTP"
        for item in data["data"]["items"]:
            assert item["type"] == "note", "should only return notes"
