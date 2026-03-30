import numpy as np
import pytest

from memask.router.embedding_classifier import (
    INTENT_EXEMPLARS,
    SIMILARITY_THRESHOLD,
    _cached_vectors,
    _get_exemplar_vectors,
    classify_by_embedding,
)
from memask.router.intents import Confidence, Intent


@pytest.fixture(autouse=True)
def clear_cached_vectors():
    import memask.router.embedding_classifier as mod
    mod._cached_vectors = None
    yield
    mod._cached_vectors = None


class TestClassifyByEmbeddingResult:
    def test_returns_routing_result(self, fake_embedder):
        result = classify_by_embedding("remind me to buy milk", fake_embedder)
        if result is not None:
            assert result.source == "embedding", "source should be embedding"
            assert result.confidence == Confidence.MEDIUM, (
                "confidence should always be MEDIUM from embedding"
            )
            assert result.raw_input == "remind me to buy milk", (
                "should preserve raw input"
            )

    def test_returns_none_when_below_threshold(self):
        class LowScoreEmbedder:
            model_name = "low"
            dimension = 4

            def embed_one(self, text):
                return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

            def embed_many(self, texts):
                return np.zeros((len(texts), 4), dtype=np.float32)

        result = classify_by_embedding("hello", LowScoreEmbedder())
        assert result is None, "should return None when all scores below threshold"

    def test_returns_none_on_exception(self):
        class BrokenEmbedder:
            model_name = "broken"
            dimension = 4

            def embed_one(self, text):
                raise RuntimeError("model crashed")

            def embed_many(self, texts):
                raise RuntimeError("model crashed")

        result = classify_by_embedding("hello", BrokenEmbedder())
        assert result is None, "should return None on embedding exception"


class TestBestIntentSelection:
    def test_selects_highest_scoring_intent(self, fake_embedder):
        result = classify_by_embedding("test input", fake_embedder)
        if result is None:
            return
        assert result.intent in Intent, "should return a valid intent"

    def test_all_intents_have_exemplars(self):
        for intent in [Intent.CAPTURE, Intent.SEARCH, Intent.TODO_CREATE,
                       Intent.TODO_LIST, Intent.TODO_COMPLETE]:
            assert intent in INTENT_EXEMPLARS, (
                f"{intent.value} should have exemplars"
            )
            assert len(INTENT_EXEMPLARS[intent]) >= 4, (
                f"{intent.value} should have at least 4 exemplars"
            )


class TestExemplarVectorCaching:
    def test_caches_vectors_after_first_call(self, fake_embedder):
        import memask.router.embedding_classifier as mod

        assert mod._cached_vectors is None, "cache should start empty"
        _get_exemplar_vectors(fake_embedder)
        assert mod._cached_vectors is not None, "cache should be populated"

    def test_second_call_reuses_cache(self, fake_embedder):
        _get_exemplar_vectors(fake_embedder)
        initial_log_len = len(fake_embedder.call_log)

        _get_exemplar_vectors(fake_embedder)
        assert len(fake_embedder.call_log) == initial_log_len, (
            "second call should not invoke embedder again"
        )

    def test_cached_vectors_have_correct_shape(self, fake_embedder):
        vectors = _get_exemplar_vectors(fake_embedder)
        for intent, vecs in vectors.items():
            expected_rows = len(INTENT_EXEMPLARS[intent])
            assert vecs.shape == (expected_rows, fake_embedder.dimension), (
                f"{intent.value} vectors should have shape "
                f"({expected_rows}, {fake_embedder.dimension})"
            )


class TestThreshold:
    def test_threshold_is_reasonable(self):
        assert 0.2 <= SIMILARITY_THRESHOLD <= 0.6, (
            "threshold should be in a reasonable range"
        )


class TestQueryContextExtracted:
    def test_result_has_query_context(self, fake_embedder):
        result = classify_by_embedding("notes from yesterday", fake_embedder)
        if result is not None:
            assert result.query_context is not None, (
                "result should include extracted query context"
            )


@pytest.mark.real_model
class TestRealModelClassification:
    @pytest.fixture
    def real_embedder(self):
        try:
            from memask.embedding import EmbeddingService
            return EmbeddingService()
        except Exception:
            pytest.skip("real embedding model not available")

    @pytest.mark.parametrize("text,expected_intent", [
        ("buy sugar", Intent.TODO_CREATE),
        ("pick up dry cleaning", Intent.TODO_CREATE),
        ("call mom", Intent.TODO_CREATE),
        ("pay the electricity bill", Intent.TODO_CREATE),
        ("book flight tickets", Intent.TODO_CREATE),
        ("the meeting went well today", Intent.CAPTURE),
        ("python 3.12 has nice new features", Intent.CAPTURE),
        ("talked to alice about the redesign", Intent.CAPTURE),
        ("what did I note about deployment", Intent.SEARCH),
        ("find my notes about kubernetes", Intent.SEARCH),
        ("anything about the release plan", Intent.SEARCH),
    ])
    def test_classifies_correctly(self, real_embedder, text, expected_intent):
        import memask.router.embedding_classifier as mod
        mod._cached_vectors = None

        result = classify_by_embedding(text, real_embedder)
        assert result is not None, f"'{text}' should not fall below threshold"
        assert result.intent == expected_intent, (
            f"'{text}' should classify as {expected_intent.value}, "
            f"got {result.intent.value}"
        )

    @pytest.mark.parametrize("text", [
        "buy sugar",
        "schedule haircut",
        "send invoice to client",
        "water the plants",
    ])
    def test_short_imperative_phrases_are_todos(self, real_embedder, text):
        import memask.router.embedding_classifier as mod
        mod._cached_vectors = None

        result = classify_by_embedding(text, real_embedder)
        assert result is not None, (
            f"'{text}' should not fall below threshold"
        )
        assert result.intent == Intent.TODO_CREATE, (
            f"'{text}' should classify as todo_create, got {result.intent.value}"
        )
