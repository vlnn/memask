import pytest

from memask.router.intents import Confidence, Intent, RoutingResult
from memask.router.query_understanding import extract_query_context
from memask.router.router import RouterConfig, route


class TestRouterConfig:
    def test_default_config(self):
        config = RouterConfig()
        assert config.embedding_threshold == Confidence.MEDIUM, (
            "should default to using embedding for MEDIUM confidence"
        )
        assert config.enable_embedding is True, (
            "embedding classification should be enabled by default"
        )

    def test_disable_embedding(self):
        config = RouterConfig(enable_embedding=False)
        result = route(
            "buy sugar",
            config=config,
        )
        assert result.source == "rules", "should use rules only when embedding disabled"


class TestRouteWithConfig:
    def test_high_confidence_ignores_embedding(self, fake_embedder):
        config = RouterConfig()
        result = route(
            "?what about deployment", embedding_service=fake_embedder, config=config
        )
        assert result.confidence == Confidence.HIGH, (
            "prefix ? should give HIGH confidence"
        )
        assert result.source == "rules", (
            "HIGH confidence should skip embedding classifier"
        )

    def test_medium_confidence_tries_embedding(self, mocker):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=None,
        )
        config = RouterConfig()

        class FakeEmb:
            pass

        route(
            "buy sugar",
            embedding_service=FakeEmb(),
            config=config,
        )
        mock_classify.assert_called_once(), (
            "MEDIUM confidence input should trigger embedding classifier"
        )

    def test_low_confidence_skips_embedding(self, mocker):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
        )
        config = RouterConfig()

        class FakeEmb:
            pass

        route(
            "deployment notes from yesterday",
            embedding_service=FakeEmb(),
            config=config,
        )
        mock_classify.assert_not_called(), (
            "LOW confidence input should not trigger embedding classifier"
        )

    def test_no_embedder_uses_rules(self):
        config = RouterConfig()
        result = route(
            "buy sugar",
            config=config,
        )
        assert result.source == "rules", (
            "should fall back to rules when no embedder provided"
        )


class TestEmbeddingOverridesRules:
    def test_embedding_can_reclassify_capture_as_todo(self, mocker):
        todo_result = RoutingResult(
            intent=Intent.TODO_CREATE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("buy sugar"),
            raw_input="buy sugar",
            source="embedding",
        )
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=todo_result,
        )

        class FakeEmb:
            pass

        result = route("buy sugar", embedding_service=FakeEmb())
        assert result.intent == Intent.TODO_CREATE, (
            "embedding should reclassify 'buy sugar' as todo"
        )
        assert result.source == "embedding", (
            "result source should be embedding not rules"
        )

    def test_embedding_can_reclassify_capture_as_search(self, mocker):
        search_result = RoutingResult(
            intent=Intent.SEARCH,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context("that deployment thing"),
            raw_input="that deployment thing",
            source="embedding",
        )
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=search_result,
        )

        class FakeEmb:
            pass

        result = route("that deployment thing", embedding_service=FakeEmb())
        assert result.intent == Intent.SEARCH, (
            "embedding should reclassify ambiguous text as search"
        )

    def test_embedding_returning_none_falls_back_to_rules(self, mocker):
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=None,
        )

        class FakeEmb:
            pass

        result = route("some random text", embedding_service=FakeEmb())
        assert result.source == "rules", (
            "should fall back to rules when embedding returns None"
        )
        assert result.intent == Intent.CAPTURE, (
            "rules fallback should classify as capture"
        )

    def test_embedding_exception_falls_back_to_rules(self, mocker):
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            side_effect=RuntimeError("model crashed"),
        )

        class FakeEmb:
            pass

        result = route("buy sugar", embedding_service=FakeEmb())
        assert result.source == "rules", (
            "should fall back to rules on embedding exception"
        )

    @pytest.mark.parametrize("text", [
        "/todo buy milk",
        "?what about deployment",
        "!help",
        "remind me to call dentist",
    ])
    def test_high_confidence_never_consults_embedding(self, mocker, text):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
        )

        class FakeEmb:
            pass

        route(text, embedding_service=FakeEmb())
        mock_classify.assert_not_called(), (
            f"HIGH confidence '{text}' should never consult embedding"
        )


class TestAmbiguousTodoInputs:
    @pytest.mark.parametrize("text", [
        "buy sugar",
        "pick up dry cleaning",
        "call mom",
        "schedule haircut",
        "pay electricity bill",
    ])
    def test_ambiguous_todos_consult_embedding(self, mocker, text):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=None,
        )

        class FakeEmb:
            pass

        route(text, embedding_service=FakeEmb())
        mock_classify.assert_called_once(), (
            f"'{text}' should consult embedding classifier"
        )

    @pytest.mark.parametrize("text", [
        "buy sugar",
        "pick up dry cleaning",
        "call mom",
    ])
    def test_ambiguous_todos_classified_by_embedding(self, mocker, text):
        todo_result = RoutingResult(
            intent=Intent.TODO_CREATE,
            confidence=Confidence.MEDIUM,
            query_context=extract_query_context(text),
            raw_input=text,
            source="embedding",
        )
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=todo_result,
        )

        class FakeEmb:
            pass

        result = route(text, embedding_service=FakeEmb())
        assert result.intent == Intent.TODO_CREATE, (
            f"'{text}' should be classified as todo by embedding"
        )
        assert result.source == "embedding", (
            f"'{text}' classification source should be embedding"
        )
