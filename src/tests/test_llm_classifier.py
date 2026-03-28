import pytest

from memask.router.intents import Confidence, Intent
from memask.router.llm_classifier import classify_by_llm, is_ollama_available


class TestOllamaAvailability:
    def test_returns_false_when_connection_refused(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._check_ollama",
            return_value=False,
        )
        assert is_ollama_available() is False, (
            "should report unavailable when connection fails"
        )

    def test_returns_true_when_reachable(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._check_ollama",
            return_value=True,
        )
        assert is_ollama_available() is True, (
            "should report available when Ollama responds"
        )


class TestClassifyByLlm:
    def test_returns_parsed_intent_from_ollama(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._call_ollama",
            return_value="search",
        )
        result = classify_by_llm("deployment notes from yesterday")
        assert result.intent == Intent.SEARCH, (
            "should parse 'search' response into Intent.SEARCH"
        )
        assert result.source == "llm", "source should be 'llm'"

    def test_returns_capture_for_unknown_response(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._call_ollama",
            return_value="gibberish_nonsense",
        )
        result = classify_by_llm("some ambiguous text")
        assert result.intent == Intent.CAPTURE, (
            "should default to capture for unparseable LLM response"
        )

    @pytest.mark.parametrize("llm_response,expected_intent", [
        ("capture", Intent.CAPTURE),
        ("search", Intent.SEARCH),
        ("todo_create", Intent.TODO_CREATE),
        ("todo_list", Intent.TODO_LIST),
        ("todo_complete", Intent.TODO_COMPLETE),
        ("app_command", Intent.APP_COMMAND),
    ])
    def test_maps_all_valid_responses(self, mocker, llm_response, expected_intent):
        mocker.patch(
            "memask.router.llm_classifier._call_ollama",
            return_value=llm_response,
        )
        result = classify_by_llm(f"test input for {llm_response}")
        assert result.intent == expected_intent, (
            f"LLM response '{llm_response}' should map to {expected_intent.value}"
        )

    def test_returns_none_when_ollama_unavailable(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._call_ollama",
            side_effect=ConnectionError("Ollama not running"),
        )
        result = classify_by_llm("test input")
        assert result is None, (
            "should return None when Ollama is unreachable"
        )

    def test_confidence_is_medium_from_llm(self, mocker):
        mocker.patch(
            "memask.router.llm_classifier._call_ollama",
            return_value="search",
        )
        result = classify_by_llm("something ambiguous")
        assert result.confidence == Confidence.MEDIUM, (
            "LLM classification should have medium confidence"
        )
