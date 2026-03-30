import os

import pytest

from memask.router.embedding_classifier import (
    INTENT_EXEMPLARS,
    classify_by_embedding,
    reset_cache,
)
from memask.router.intents import Intent

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


_skip_no_model = pytest.mark.skipif(
    not _model_is_cached(),
    reason=(
        f"model {MODEL_NAME} not cached locally, run: "
        f"python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL_NAME}')\""
    ),
)


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    reset_cache()
    yield
    reset_cache()


class TestCaptureExemplarsPresent:
    def test_has_operational_exemplars(self):
        capture_texts = " ".join(INTENT_EXEMPLARS[Intent.CAPTURE]).lower()
        assert "restart" in capture_texts or "heavy load" in capture_texts, (
            "CAPTURE exemplars should include operational/infra status language"
        )

    def test_has_past_tense_action_exemplars(self):
        capture_texts = " ".join(INTENT_EXEMPLARS[Intent.CAPTURE]).lower()
        assert "had to" in capture_texts or "crashed" in capture_texts, (
            "CAPTURE exemplars should include past-tense action reports"
        )


@pytest.mark.real_model
class TestOperationalNotesClassifyAsCapture:
    @_skip_no_model
    @pytest.mark.parametrize("text", [
        "dirty is under heavy load after restart -- and I had to restart it due to mullvad update/network stall",
        "the server crashed twice this morning",
        "had to restart nginx after the config change",
        "redis is using 90% of available memory",
        "switched to backup DNS because primary was flapping",
        "the CI pipeline broke after the ubuntu upgrade",
    ])
    def test_operational_status_routes_to_capture(self, text):
        embedder = _load_real_embedder()
        result = classify_by_embedding(text, embedder)
        assert result is None or result.intent == Intent.CAPTURE, (
            f"'{text[:60]}...' should classify as CAPTURE, got {result.intent.value if result else 'None'}"
        )

    @_skip_no_model
    @pytest.mark.parametrize("text", [
        "remind me to buy milk",
        "need to fix the auth bug tomorrow",
        "don't forget to email the client",
    ])
    def test_actual_todos_still_route_to_todo_create(self, text):
        embedder = _load_real_embedder()
        result = classify_by_embedding(text, embedder)
        assert result is not None, (
            f"'{text}' should not fall below threshold"
        )
        assert result.intent == Intent.TODO_CREATE, (
            f"'{text}' should still classify as TODO_CREATE"
        )

    @_skip_no_model
    @pytest.mark.parametrize("text", [
        "the meeting went well today",
        "python 3.12 has nice new features",
        "decided to use postgres instead of mysql",
    ])
    def test_original_capture_exemplars_still_work(self, text):
        embedder = _load_real_embedder()
        result = classify_by_embedding(text, embedder)
        assert result is None or result.intent == Intent.CAPTURE, (
            f"'{text}' should still classify as CAPTURE"
        )
