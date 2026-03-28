import pytest

from memask.router.intents import Confidence, Intent
from memask.router.router import route


class TestRulesFirst:
    @pytest.mark.parametrize("text,expected_intent", [
        ("/todo buy milk", Intent.TODO_CREATE),
        ("/todo list", Intent.TODO_LIST),
        ("?what about deployment", Intent.SEARCH),
        ("!help", Intent.APP_COMMAND),
        ("remind me to buy milk", Intent.TODO_CREATE),
    ])
    def test_high_confidence_rules_skip_llm(self, mocker, text, expected_intent):
        llm_spy = mocker.patch(
            "memask.router.router.classify_by_llm",
        )
        result = route(text)
        assert result.intent == expected_intent, (
            f"'{text}' should route to {expected_intent.value} via rules"
        )
        llm_spy.assert_not_called()


class TestLlmFallback:
    def test_ambiguous_input_triggers_llm(self, mocker):
        mocker.patch(
            "memask.router.router.is_ollama_available",
            return_value=True,
        )
        mock_llm = mocker.patch(
            "memask.router.router.classify_by_llm",
            return_value=mocker.MagicMock(
                intent=Intent.SEARCH,
                confidence=Confidence.MEDIUM,
                source="llm",
                raw_input="deployment notes from yesterday",
                query_context=mocker.MagicMock(),
            ),
        )
        result = route("deployment notes from yesterday")
        mock_llm.assert_called_once()
        assert result.intent == Intent.SEARCH, (
            "ambiguous input should use LLM classification"
        )
        assert result.source == "llm", "source should be 'llm' when LLM used"

    def test_medium_confidence_still_uses_rules(self, mocker):
        mocker.patch(
            "memask.router.router.is_ollama_available",
            return_value=False,
        )
        result = route("i need to update the docs")
        assert result.intent == Intent.TODO_CREATE, (
            "medium confidence should use rules when LLM unavailable"
        )
        assert result.source == "rules", "source should be 'rules'"


class TestGracefulDegradation:
    def test_ollama_down_uses_rules_only(self, mocker):
        mocker.patch(
            "memask.router.router.is_ollama_available",
            return_value=False,
        )
        llm_spy = mocker.patch("memask.router.router.classify_by_llm")
        result = route("deployment notes from yesterday")
        llm_spy.assert_not_called()
        assert result.source == "rules", (
            "should use rules when Ollama is unavailable"
        )

    def test_llm_returns_none_falls_back_to_rules(self, mocker):
        mocker.patch(
            "memask.router.router.is_ollama_available",
            return_value=True,
        )
        mocker.patch(
            "memask.router.router.classify_by_llm",
            return_value=None,
        )
        result = route("deployment notes from yesterday")
        assert result.source == "rules", (
            "should fall back to rules when LLM returns None"
        )

    def test_never_blocks_on_llm_failure(self, mocker):
        mocker.patch(
            "memask.router.router.is_ollama_available",
            return_value=True,
        )
        mocker.patch(
            "memask.router.router.classify_by_llm",
            side_effect=Exception("unexpected LLM error"),
        )
        result = route("deployment notes from yesterday")
        assert result is not None, "should never fail even if LLM raises"
        assert result.source == "rules", "should fall back to rules on LLM error"


class TestQueryContextPassthrough:
    def test_search_result_includes_query_context(self):
        result = route("?what about deployment yesterday")
        assert result.query_context is not None, (
            "search result should include query context"
        )
        assert result.query_context.topic == "deployment", (
            "should extract topic from search query"
        )

    def test_capture_result_includes_raw_input(self):
        result = route("kubernetes cluster needs more RAM")
        assert result.raw_input == "kubernetes cluster needs more RAM", (
            "capture result should preserve raw input"
        )
