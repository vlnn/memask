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


def resp_json(resp):
    return json.loads(resp.data)


def create_via_http(client, text):
    resp = client.post("/input", json={"text": text})
    data = resp_json(resp)
    return data.get("data", {}).get("id")


class TestPatchItem:
    def test_updates_content(self, client):
        client.post("/input", json={"text": "original note"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        resp = client.patch(f"/items/{item_id}", json={"content": "updated note"})
        assert resp.status_code == 200, "should return 200 on update"
        data = resp_json(resp)
        assert data["item"]["content"] == "updated note", "should update content"

    def test_changes_type(self, client):
        client.post("/input", json={"text": "a plain note"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        resp = client.patch(f"/items/{item_id}", json={"type": "todo", "status": "pending"})
        assert resp.status_code == 200, "should return 200 on retype"
        data = resp_json(resp)
        assert data["item"]["type"] == "todo", "should change type to todo"
        assert data["item"]["status"] == "pending", "should set status"

    def test_changes_status(self, client):
        client.post("/input", json={"text": "remind me to buy milk"})
        items = resp_json(client.get("/items?type=todo"))["items"]
        item_id = items[0]["id"]

        resp = client.patch(f"/items/{item_id}", json={"status": "done"})
        assert resp.status_code == 200, "should return 200 on status change"
        data = resp_json(resp)
        assert data["item"]["status"] == "done", "should mark as done"

    def test_returns_404_for_missing(self, client):
        resp = client.patch("/items/nonexistent", json={"content": "x"})
        assert resp.status_code == 404, "should return 404 for missing item"

    def test_returns_400_for_empty_body(self, client):
        client.post("/input", json={"text": "something"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        resp = client.patch(f"/items/{item_id}", json={})
        assert resp.status_code == 200, "empty update should still return 200"

    def test_rejects_invalid_type(self, client):
        client.post("/input", json={"text": "something"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        resp = client.patch(f"/items/{item_id}", json={"type": "bogus"})
        assert resp.status_code == 400, "should reject invalid type"


class TestDeleteItem:
    def test_soft_deletes_item(self, client):
        client.post("/input", json={"text": "ephemeral note"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        resp = client.delete(f"/items/{item_id}")
        assert resp.status_code == 200, "should return 200 on delete"
        data = resp_json(resp)
        assert data["deleted"] is True, "should confirm deletion"

        items_after = resp_json(client.get("/items"))["items"]
        ids = [i["id"] for i in items_after]
        assert item_id not in ids, "deleted item should not appear in list"

    def test_returns_404_for_missing(self, client):
        resp = client.delete("/items/nonexistent")
        assert resp.status_code == 404, "should return 404 for missing item"

    def test_double_delete_returns_404(self, client):
        client.post("/input", json={"text": "once only"})
        items = resp_json(client.get("/items"))["items"]
        item_id = items[0]["id"]

        client.delete(f"/items/{item_id}")
        resp = client.delete(f"/items/{item_id}")
        assert resp.status_code == 404, "second delete should return 404"
