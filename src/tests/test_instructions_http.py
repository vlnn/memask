import json

import pytest

from memask.app import AppContext
from memask.server import create_app
from tests.helpers import FakeEmbeddingService, FakeLLM, FakeReranker


def resp_json(resp):
    return json.loads(resp.data)


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


class TestInstructionsHTTPEndpoints:
    def test_create_via_input(self, client):
        resp = client.post("/input", json={"text": "/instruction be brief"})
        data = resp_json(resp)
        assert data["action"] == "instruction_saved", (
            "POST /input with /instruction should save instruction"
        )

    def test_list_via_input(self, client):
        client.post("/input", json={"text": "/instruction be brief"})
        resp = client.post("/input", json={"text": "/instruction"})
        data = resp_json(resp)
        assert data["action"] == "instruction_listed", (
            "POST /input with bare /instruction should list"
        )
        assert len(data["data"]["instructions"]) == 1, (
            "should list the one saved instruction"
        )

    def test_get_instructions(self, client):
        client.post("/input", json={"text": "/instruction rule one"})
        client.post("/input", json={"text": "/instruction rule two"})
        resp = client.get("/instructions")
        data = resp_json(resp)
        assert len(data) == 2, (
            "GET /instructions should return active instructions"
        )

    def test_delete_instruction(self, client):
        client.post("/input", json={"text": "/instruction delete me"})
        instructions = resp_json(client.get("/instructions"))
        inst_id = instructions[0]["id"]
        resp = client.delete(f"/instructions/{inst_id}")
        assert resp.status_code == 200, (
            "DELETE /instructions/:id should return 200"
        )
        remaining = resp_json(client.get("/instructions"))
        assert len(remaining) == 0, (
            "deleted instruction should no longer appear"
        )

    def test_delete_unknown_instruction(self, client):
        resp = client.delete("/instructions/nonexistent")
        assert resp.status_code == 404, (
            "DELETE unknown id should return 404"
        )

    def test_clear_via_input(self, client):
        client.post("/input", json={"text": "/instruction one"})
        client.post("/input", json={"text": "/instruction two"})
        resp = client.post("/input", json={"text": "/instruction clear"})
        data = resp_json(resp)
        assert data["action"] == "instruction_cleared", (
            "should return instruction_cleared"
        )
        remaining = resp_json(client.get("/instructions"))
        assert len(remaining) == 0, (
            "all instructions should be cleared"
        )
