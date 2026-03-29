import os

import pytest

from tests.helpers import FakeEmbeddingService
from memask.repository.items import create_item, soft_delete_item
from memask.search.hybrid import hybrid_search
from memask.search.keyword import keyword_search
from memask.search.startup import on_startup
from memask.search.vector_store import VectorStore
from memask.search.worker import enqueue_embedding, process_all_pending

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "all-MiniLM-L6-v2"


def _model_is_cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(f"sentence-transformers/{MODEL_NAME}", "config.json")
        return result is not None and isinstance(result, str)
    except Exception:
        return False


def _load_real_embedder():
    from memask.search.embedding import EmbeddingService
    return EmbeddingService(MODEL_NAME)


@pytest.fixture
def store(lance_dir):
    return VectorStore(lance_dir)


def _create_and_index(conn, store, embedder, content, **kwargs):
    item = create_item(conn, content, **kwargs)
    enqueue_embedding(conn, item.id)
    process_all_pending(conn, store, embedder)
    return item


class TestSemanticSimilarity:
    """search 'deployment' returns results even if the note says 'shipping to production'"""

    @pytest.mark.real_model
    @pytest.mark.skipif(
        not _model_is_cached(),
        reason=(
            f"model {MODEL_NAME} not cached locally, run: "
            f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL_NAME}')\""
        ),
    )
    def test_semantic_finds_related_content(self, conn, store):
        embedder = _load_real_embedder()

        _create_and_index(conn, store, embedder, "shipping to production on friday")
        _create_and_index(conn, store, embedder, "grocery list: eggs, milk, bread")
        _create_and_index(conn, store, embedder, "release pipeline and deployment steps")

        results = hybrid_search(conn, store, embedder, "that thing about deployment")
        contents = [r.item.content for r in results]

        assert any("shipping to production" in c for c in contents), (
            "semantic search should find 'shipping to production' when querying 'deployment'"
        )
        assert any("deployment steps" in c for c in contents), (
            "should also find the note with literal 'deployment'"
        )

    def test_keyword_finds_literal_match(self, conn, store, fake_embedder):
        _create_and_index(conn, store, fake_embedder, "deployment checklist for v2")
        _create_and_index(conn, store, fake_embedder, "grocery list: eggs, milk")

        results = keyword_search(conn, "deployment")
        assert len(results) == 1, "keyword search should find exact match"
        assert "deployment" in results[0].item.content, "should match literal word"

    def test_keyword_misses_semantic_match(self, conn, store, fake_embedder):
        _create_and_index(conn, store, fake_embedder, "shipping to production on friday")

        results = keyword_search(conn, "deployment")
        assert len(results) == 0, "keyword search should NOT find 'shipping to production' for query 'deployment'"


class TestReconciliationEndToEnd:
    def test_orphaned_vectors_cleaned_on_startup(self, conn, store, fake_embedder):
        item_a = _create_and_index(conn, store, fake_embedder, "keep this")
        item_b = _create_and_index(conn, store, fake_embedder, "delete this")
        assert store.count() == 2, "both items should be indexed"

        soft_delete_item(conn, item_b.id)

        on_startup(conn, store, fake_embedder)
        assert store.list_item_ids() == {item_a.id}, "only active item should remain after startup"

    def test_deleted_items_excluded_from_search(self, conn, store, fake_embedder):
        item = _create_and_index(conn, store, fake_embedder, "secret note about passwords")
        soft_delete_item(conn, item.id)

        on_startup(conn, store, fake_embedder)

        results = hybrid_search(conn, store, fake_embedder, "passwords")
        assert len(results) == 0, "deleted items should not appear in search results"


class TestModelVersionReindexEndToEnd:
    def test_model_change_triggers_reindex(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")
        item = _create_and_index(conn, store, old_embedder, "important note")
        assert store.stale_items(fake_embedder.model_name) == {item.id}, (
            "item should be stale under new model"
        )

        on_startup(conn, store, fake_embedder)
        process_all_pending(conn, store, fake_embedder)

        assert store.stale_items(fake_embedder.model_name) == set(), (
            "no items should be stale after startup + processing"
        )

    def test_mixed_models_converge_after_startup(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")

        _create_and_index(conn, store, old_embedder, "indexed with old model")
        _create_and_index(conn, store, fake_embedder, "indexed with new model")

        on_startup(conn, store, fake_embedder)
        process_all_pending(conn, store, fake_embedder)

        stale = store.stale_items(fake_embedder.model_name)
        assert stale == set(), "all items should use current model after full reindex"

    def test_search_works_during_partial_reindex(self, conn, store, fake_embedder):
        old_embedder = FakeEmbeddingService(model_name="old-model")

        _create_and_index(conn, store, old_embedder, "stale note about cats")
        _create_and_index(conn, store, fake_embedder, "fresh note about dogs")

        results = hybrid_search(conn, store, fake_embedder, "dogs")
        assert len(results) >= 1, "search should still work with mixed model vectors"
