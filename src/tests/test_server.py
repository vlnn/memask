import json

import pytest

from memask.app import AppContext
from memask.server import create_app
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


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200, "health should return 200"

    def test_reports_status_ok(self, client):
        data = resp_json(client.get("/health"))
        assert data["status"] == "ok", "health status should be ok"

    def test_reports_llm_availability(self, client):
        data = resp_json(client.get("/health"))
        assert "llm" in data, "health should report llm status"
        assert data["llm"]["available"] is True, "fake llm should be available"

    def test_reports_job_queue(self, client):
        data = resp_json(client.get("/health"))
        assert "jobs" in data, "health should report job queue stats"

    def test_reports_embedder_info(self, client):
        data = resp_json(client.get("/health"))
        assert "embedder" in data, "health should report embedder info"
        assert data["embedder"]["model"] == "test-model", (
            "health should report embedder model name"
        )


class TestHealthWithoutLLM:
    def test_llm_unavailable(self, tmp_path):
        ctx = AppContext(
            db_path=str(tmp_path / "test.db"),
            llm_factory=lambda: FakeLLM(available=False),
        )
        try:
            app = create_app(ctx)
            app.config["TESTING"] = True
            with app.test_client() as c:
                data = resp_json(c.get("/health"))
                assert data["llm"]["available"] is False, (
                    "should report llm unavailable"
                )
        finally:
            ctx.shutdown()


class TestInputEndpoint:
    def test_capture_note(self, client):
        resp = client.post("/input", json={"text": "meeting went well"})
        assert resp.status_code == 200, "input should return 200"
        data = resp_json(resp)
        assert data["action"] == "captured", "plain text should be captured"

    def test_search_query(self, client):
        client.post("/input", json={"text": "deployment notes"})
        resp = client.post("/input", json={"text": "?deployment"})
        data = resp_json(resp)
        assert data["action"] in ("searched", "answered"), (
            "question should trigger search or answer"
        )

    def test_todo_create(self, client):
        resp = client.post("/input", json={"text": "remind me to buy milk"})
        data = resp_json(resp)
        assert data["action"] == "todo_created", "should create todo"

    def test_todo_list(self, client):
        client.post("/input", json={"text": "remind me to buy milk"})
        resp = client.post("/input", json={"text": "/todo list"})
        data = resp_json(resp)
        assert data["action"] == "todo_listed", "should list todos"

    def test_missing_text_returns_400(self, client):
        resp = client.post("/input", json={})
        assert resp.status_code == 400, "missing text should return 400"

    def test_empty_text_returns_400(self, client):
        resp = client.post("/input", json={"text": ""})
        assert resp.status_code == 400, "empty text should return 400"

    def test_no_json_body_returns_400(self, client):
        resp = client.post("/input", data="plain text")
        assert resp.status_code == 400, "non-json body should return 400"


class TestItemsEndpoint:
    def test_lists_items(self, client):
        client.post("/input", json={"text": "first note"})
        client.post("/input", json={"text": "second note"})
        resp = client.get("/items")
        assert resp.status_code == 200, "items should return 200"
        data = resp_json(resp)
        assert len(data["items"]) >= 2, "should list created items"

    def test_filters_by_type(self, client):
        client.post("/input", json={"text": "plain note"})
        client.post("/input", json={"text": "remind me to buy milk"})
        resp = client.get("/items?type=todo")
        data = resp_json(resp)
        for item in data["items"]:
            assert item["type"] == "todo", "should only return todos"

    def test_empty_list(self, client):
        resp = client.get("/items")
        data = resp_json(resp)
        assert data["items"] == [], "empty db should return empty list"


class TestSearchEndpoint:
    def test_search_returns_results(self, client):
        client.post("/input", json={"text": "deployment pipeline is ready"})
        resp = client.get("/search?q=deployment")
        assert resp.status_code == 200, "search should return 200"
        data = resp_json(resp)
        assert len(data["results"]) > 0, "should find matching items"

    def test_search_requires_query(self, client):
        resp = client.get("/search")
        assert resp.status_code == 400, "missing query should return 400"

    def test_search_empty_query_returns_400(self, client):
        resp = client.get("/search?q=")
        assert resp.status_code == 400, "empty query should return 400"

    def test_search_no_results(self, client):
        resp = client.get("/search?q=nonexistent")
        data = resp_json(resp)
        assert data["results"] == [], "no matches should return empty list"


class TestSettingsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200, "settings should return 200"

    def test_returns_dict(self, client):
        data = resp_json(client.get("/settings"))
        assert isinstance(data, dict), "settings should return a dict"


class TestJsonContentType:
    @pytest.mark.parametrize("endpoint", ["/health", "/items", "/settings"])
    def test_get_endpoints_return_json(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.content_type == "application/json", (
            f"{endpoint} should return application/json"
        )

    def test_input_returns_json(self, client):
        resp = client.post("/input", json={"text": "hello"})
        assert resp.content_type == "application/json", (
            "input should return application/json"
        )


def resp_json(resp):
    return json.loads(resp.data)
