import numpy as np
import pytest

from memask.search.vector_store import VectorStore


def _random_vector(dim=384, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestVectorStoreReopenExistingTable:
    def test_second_instance_opens_existing_table(self, lance_dir):
        store1 = VectorStore(lance_dir)
        store1.add("item-1", 0, "hello", _random_vector(seed=1), "test-model")
        assert store1.count() == 1, "first store should have one record"

        store2 = VectorStore(lance_dir)
        assert store2.count() == 1, (
            "second store should see data from first"
        )

    def test_second_instance_can_add(self, lance_dir):
        store1 = VectorStore(lance_dir)
        store1.add("item-1", 0, "first", _random_vector(seed=1), "test-model")

        store2 = VectorStore(lance_dir)
        store2.add("item-2", 0, "second", _random_vector(seed=2), "test-model")
        assert store2.count() == 2, (
            "second store should add to existing table"
        )

    def test_second_instance_can_search(self, lance_dir):
        vec = _random_vector(seed=1)
        store1 = VectorStore(lance_dir)
        store1.add("item-1", 0, "hello", vec, "test-model")

        store2 = VectorStore(lance_dir)
        results = store2.search(vec, limit=5)
        assert len(results) == 1, (
            "second store should find vectors from first"
        )
        assert results[0]["item_id"] == "item-1", (
            "should return correct item from reopened table"
        )

    def test_survives_multiple_reopens(self, lance_dir):
        for i in range(5):
            store = VectorStore(lance_dir)
            store.add(f"item-{i}", 0, f"text-{i}", _random_vector(seed=i), "test-model")

        final = VectorStore(lance_dir)
        assert final.count() == 5, (
            "should accumulate data across multiple reopens"
        )

    def test_empty_reopen_then_add(self, lance_dir):
        store1 = VectorStore(lance_dir)
        assert store1.count() == 0, "fresh store should be empty"

        store2 = VectorStore(lance_dir)
        store2.add("item-1", 0, "hello", _random_vector(), "test-model")
        assert store2.count() == 1, (
            "should add to table created by previous empty instance"
        )
