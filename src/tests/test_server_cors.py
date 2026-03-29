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


class TestCorsHeaders:
    @pytest.mark.parametrize("endpoint", [
        "/health",
        "/items",
        "/settings",
        "/search?q=test",
    ])
    def test_get_endpoints_include_cors_headers(self, client, endpoint):
        resp = client.get(endpoint)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
            f"{endpoint} should include CORS allow-origin header"
        )

    def test_post_input_includes_cors_headers(self, client):
        resp = client.post("/input", json={"text": "hello"})
        assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
            "/input POST should include CORS allow-origin header"
        )

    def test_error_responses_include_cors_headers(self, client):
        resp = client.post("/input", json={})
        assert resp.status_code == 400, "missing text should return 400"
        assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
            "error responses should also include CORS headers"
        )

    @pytest.mark.parametrize("endpoint", [
        "/health",
        "/input",
        "/items",
        "/search",
        "/settings",
    ])
    def test_options_preflight_returns_cors_headers(self, client, endpoint):
        resp = client.options(endpoint)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*", (
            f"OPTIONS {endpoint} should include allow-origin"
        )
        assert "Content-Type" in resp.headers.get("Access-Control-Allow-Headers", ""), (
            f"OPTIONS {endpoint} should allow Content-Type header"
        )
