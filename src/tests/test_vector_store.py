import numpy as np
import pytest

from memask.search.vector_store import VectorStore


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


def _random_vector(dim=384, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestVectorStoreAdd:
    def test_add_and_count(self, store):
        store.add("item-1", 0, "hello", _random_vector(), "test-model")
        assert store.count() == 1, "should have one record after add"

    def test_add_batch(self, store):
        records = [
            {
                "item_id": f"item-{i}",
                "chunk_index": 0,
                "text": f"text {i}",
                "model_version": "test-model",
                "vector": _random_vector(seed=i).tolist(),
            }
            for i in range(5)
        ]
        store.add_batch(records)
        assert store.count() == 5, "should have all batch records"

    def test_add_batch_empty(self, store):
        store.add_batch([])
        assert store.count() == 0, "empty batch should be a no-op"


class TestVectorStoreSearch:
    def test_search_returns_results(self, store):
        vec = _random_vector(seed=1)
        store.add("item-1", 0, "hello world", vec, "test-model")
        results = store.search(vec, limit=5)
        assert len(results) == 1, "should find the stored vector"
        assert results[0]["item_id"] == "item-1", "should return correct item id"

    def test_search_returns_nearest(self, store):
        target = _random_vector(seed=10)
        close = target + np.random.RandomState(99).randn(384).astype(np.float32) * 0.01
        close = close / np.linalg.norm(close)
        far = _random_vector(seed=999)

        store.add("close-item", 0, "close", close, "test-model")
        store.add("far-item", 0, "far", far, "test-model")

        results = store.search(target, limit=2)
        assert results[0]["item_id"] == "close-item", "nearest vector should rank first"

    def test_search_filters_by_model(self, store):
        vec = _random_vector(seed=1)
        store.add("item-old", 0, "old", vec, "model-v1")
        store.add("item-new", 0, "new", vec, "model-v2")

        results = store.search(vec, limit=10, model_version="model-v2")
        ids = {r["item_id"] for r in results}
        assert "item-new" in ids, "should include matching model version"
        assert "item-old" not in ids, "should exclude other model versions"

    def test_search_respects_limit(self, store):
        for i in range(10):
            store.add(f"item-{i}", 0, f"text {i}", _random_vector(seed=i), "test-model")
        results = store.search(_random_vector(seed=0), limit=3)
        assert len(results) == 3, "should respect limit"


class TestVectorStoreDelete:
    def test_delete_by_item(self, store):
        store.add("item-1", 0, "chunk 0", _random_vector(seed=1), "test-model")
        store.add("item-1", 1, "chunk 1", _random_vector(seed=2), "test-model")
        store.add("item-2", 0, "other", _random_vector(seed=3), "test-model")

        store.delete_by_item("item-1")
        assert store.count() == 1, "should delete all chunks for item"
        ids = store.list_item_ids()
        assert "item-1" not in ids, "deleted item should not be listed"
        assert "item-2" in ids, "other items should remain"

    def test_delete_by_model(self, store):
        store.add("item-1", 0, "v1", _random_vector(seed=1), "model-v1")
        store.add("item-2", 0, "v2", _random_vector(seed=2), "model-v2")

        store.delete_by_model("model-v1")
        ids = store.list_item_ids()
        assert "item-1" not in ids, "should delete records for old model"
        assert "item-2" in ids, "should keep records for current model"


class TestVectorStoreListAndStale:
    def test_list_item_ids(self, store):
        store.add("item-a", 0, "a", _random_vector(seed=1), "test-model")
        store.add("item-b", 0, "b", _random_vector(seed=2), "test-model")
        store.add("item-a", 1, "a chunk 2", _random_vector(seed=3), "test-model")

        ids = store.list_item_ids()
        assert ids == {"item-a", "item-b"}, "should return unique item ids"

    def test_list_item_ids_empty(self, store):
        ids = store.list_item_ids()
        assert ids == set(), "empty store should return empty set"

    def test_stale_items(self, store):
        store.add("item-1", 0, "old", _random_vector(seed=1), "model-v1")
        store.add("item-2", 0, "new", _random_vector(seed=2), "model-v2")

        stale = store.stale_items("model-v2")
        assert stale == {"item-1"}, "items with old model version should be stale"

    def test_no_stale_when_all_current(self, store):
        store.add("item-1", 0, "current", _random_vector(seed=1), "model-v2")
        stale = store.stale_items("model-v2")
        assert stale == set(), "no items should be stale when all use current model"
