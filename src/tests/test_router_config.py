from memask.router.intents import Confidence
from memask.router.router import RouterConfig, route


class TestRouterConfig:
    def test_default_config(self):
        config = RouterConfig()
        assert config.embedding_threshold == Confidence.LOW, (
            "should default to using embedding only for LOW confidence"
        )
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
        mock_classify.assert_called_once()

    def test_no_embedder_uses_rules(self):
        config = RouterConfig()
        result = route(
            "deployment notes from yesterday",
            config=config,
        )
        assert result.source == "rules", (
            "should fall back to rules when no embedder provided"
        )
