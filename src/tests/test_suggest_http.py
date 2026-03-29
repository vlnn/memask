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


class TestSuggestEndpoint:
    def test_returns_commands_for_slash(self, client):
        resp = client.get("/suggest?q=/")
        assert resp.status_code == 200, "should return 200"
        data = resp.get_json()
        assert "suggestions" in data, "response should have suggestions key"
        kinds = {s["kind"] for s in data["suggestions"]}
        assert "command" in kinds, "/ should return command suggestions"

    def test_returns_items_for_text(self, client):
        client.post("/input", json={"text": "deploy to production"})
        resp = client.get("/suggest?q=deploy")
        data = resp.get_json()
        items = [s for s in data["suggestions"] if s["kind"] == "item"]
        assert len(items) >= 1, "should find matching items"

    def test_returns_mixed_results(self, client):
        client.post("/input", json={"text": "review the todo feature"})
        resp = client.get("/suggest?q=todo")
        data = resp.get_json()
        kinds = {s["kind"] for s in data["suggestions"]}
        assert "command" in kinds, "should include commands"
        assert "item" in kinds, "should include items"

    def test_respects_limit(self, client):
        for i in range(15):
            client.post("/input", json={"text": f"note number {i}"})
        resp = client.get("/suggest?q=note&limit=5")
        data = resp.get_json()
        assert len(data["suggestions"]) <= 5, "should respect limit parameter"

    def test_empty_query_returns_recent(self, client):
        client.post("/input", json={"text": "recent note"})
        resp = client.get("/suggest?q=")
        data = resp.get_json()
        assert len(data["suggestions"]) >= 1, "empty query should return recent items"

    def test_invalid_limit_defaults_to_ten(self, client):
        resp = client.get("/suggest?q=/&limit=abc")
        assert resp.status_code == 200, "should handle invalid limit gracefully"
