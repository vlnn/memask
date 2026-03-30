import pytest

from memask.router.intents import Confidence, Intent
from memask.router.router import RouterConfig, route


class TestRouterConfig:
    def test_default_config(self):
        config = RouterConfig()
        assert config.enable_embedding is True, (
            "embedding classification should be enabled by default"
        )

    def test_disable_embedding(self):
        config = RouterConfig(enable_embedding=False)
        result = route(
            "deployment notes from yesterday",
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
            "buy contact lenses again",
            embedding_service=FakeEmb(),
            config=config,
        )
        mock_classify.assert_called_once(), (
            "MEDIUM confidence CAPTURE should consult embedding classifier"
        )

    def test_low_confidence_tries_embedding(self, mocker):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=None,
        )
        config = RouterConfig()

        class FakeEmb:
            pass

        route(
            "deployment notes from yesterday",
            embedding_service=FakeEmb(),
            config=config,
        )
        mock_classify.assert_called_once(), (
            "LOW confidence should consult embedding classifier"
        )

    def test_no_embedder_uses_rules(self):
        config = RouterConfig()
        result = route(
            "deployment notes from yesterday",
            config=config,
        )
        assert result.source == "rules", (
            "should fall back to rules when no embedder provided"
        )

    @pytest.mark.parametrize("text", [
        "/todo buy milk",
        "?what about deployment",
        "!help",
        "remind me to buy milk",
    ])
    def test_high_confidence_rules_skip_embedding(self, mocker, text):
        mock_classify = mocker.patch(
            "memask.router.router.classify_by_embedding",
        )
        config = RouterConfig()

        class FakeEmb:
            pass

        route(text, embedding_service=FakeEmb(), config=config)
        mock_classify.assert_not_called(), (
            f"HIGH confidence '{text}' should not consult embedding"
        )

    def test_embedding_result_overrides_rules(self, mocker):
        from memask.router.intents import RoutingResult, QueryContext
        emb_result = RoutingResult(
            intent=Intent.TODO_CREATE,
            confidence=Confidence.MEDIUM,
            query_context=QueryContext(raw_query="buy a book"),
            raw_input="buy a book",
            source="embedding",
        )
        mocker.patch(
            "memask.router.router.classify_by_embedding",
            return_value=emb_result,
        )
        config = RouterConfig()

        class FakeEmb:
            pass

        result = route("buy a book", embedding_service=FakeEmb(), config=config)
        assert result.intent == Intent.TODO_CREATE, (
            "embedding result should override rules CAPTURE fallback"
        )
        assert result.source == "embedding", (
            "should report embedding as the classification source"
        )
