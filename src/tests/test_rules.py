import pytest

from memask.router.intents import Confidence, Intent
from memask.router.rules import classify_by_rules


class TestPrefixPatterns:
    @pytest.mark.parametrize("text,expected_intent", [
        ("/todo buy milk", Intent.TODO_CREATE),
        ("/todo add fix the faucet", Intent.TODO_CREATE),
        ("/todo list", Intent.TODO_LIST),
        ("/todo ls", Intent.TODO_LIST),
        ("/todo done buy milk", Intent.TODO_COMPLETE),
        ("/todo complete fix faucet", Intent.TODO_COMPLETE),
        ("!help", Intent.APP_COMMAND),
        ("!settings", Intent.APP_COMMAND),
        ("!quit", Intent.APP_COMMAND),
        ("!status", Intent.APP_COMMAND),
    ])
    def test_prefix_routes_to_correct_intent(self, text, expected_intent):
        result = classify_by_rules(text)
        assert result.intent == expected_intent, (
            f"'{text}' should route to {expected_intent.value}"
        )

    @pytest.mark.parametrize("text", [
        "/todo buy milk",
        "/todo list",
        "!help",
        "?what is python",
    ])
    def test_prefix_patterns_have_high_confidence(self, text):
        result = classify_by_rules(text)
        assert result.confidence == Confidence.HIGH, (
            f"prefix pattern '{text}' should have high confidence"
        )


class TestQuestionPatterns:
    @pytest.mark.parametrize("text", [
        "?what did I note about deployment",
        "?find my notes on kubernetes",
        "what did I note about deployment?",
        "what were my notes on the release?",
        "when did I write about testing?",
        "how did I set up the database?",
        "where did I put the config notes?",
        "who suggested the new architecture?",
        "which framework did I choose?",
        "find my notes about python",
        "search for deployment notes",
        "look up my meeting notes",
        "show me notes about kubernetes",
    ])
    def test_question_patterns_route_to_search(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.SEARCH, (
            f"'{text}' should route to search"
        )


class TestTodoCreatePatterns:
    @pytest.mark.parametrize("text", [
        "remind me to buy milk",
        "remind me to call the dentist",
        "remind about reading",
        "remind about reading book",
        "remind me about the meeting",
        "todo buy groceries",
        "todo: fix the bug in auth",
        "add todo review PR",
        "i need to finish the report",
        "don't forget to send the email",
        "remember to water the plants",
    ])
    def test_todo_triggers_route_to_todo_create(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.TODO_CREATE, (
            f"'{text}' should route to todo_create"
        )


class TestCapturePatterns:
    @pytest.mark.parametrize("text", [
        "the meeting went well today",
        "kubernetes cluster needs more RAM",
        "talked to alice about the redesign",
        "python 3.12 has nice new features",
        "just learned about lancedb for vector search",
        "\u043d\u043e\u0442\u0430\u0442\u043a\u0438 \u0437\u0456 \u0437\u0443\u0441\u0442\u0440\u0456\u0447\u0456 \u0437 \u043a\u043e\u043c\u0430\u043d\u0434\u043e\u044e",
    ])
    def test_plain_statements_route_to_capture(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.CAPTURE, (
            f"'{text}' should route to capture"
        )


class TestAppCommandPatterns:
    @pytest.mark.parametrize("text", [
        "/help",
        "/settings",
        "/quit",
        "/status",
        "/export",
    ])
    def test_slash_commands_route_to_app_command(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.APP_COMMAND, (
            f"'{text}' should route to app_command"
        )


class TestConfidenceLevels:
    @pytest.mark.parametrize("text,expected_confidence", [
        ("/todo buy milk", Confidence.HIGH),
        ("?what about deployment", Confidence.HIGH),
        ("!help", Confidence.HIGH),
        ("remind me to buy milk", Confidence.HIGH),
        ("find my notes about python", Confidence.HIGH),
        ("what did I write about testing?", Confidence.HIGH),
        ("i need to update the docs", Confidence.MEDIUM),
        ("deployment notes from yesterday", Confidence.LOW),
    ])
    def test_confidence_reflects_pattern_strength(self, text, expected_confidence):
        result = classify_by_rules(text)
        assert result.confidence == expected_confidence, (
            f"'{text}' should have {expected_confidence.value} confidence"
        )


class TestEdgeCases:
    def test_empty_string_defaults_to_capture(self):
        result = classify_by_rules("")
        assert result.intent == Intent.CAPTURE, "empty input should default to capture"

    def test_whitespace_only_defaults_to_capture(self):
        result = classify_by_rules("   ")
        assert result.intent == Intent.CAPTURE, "whitespace should default to capture"

    def test_preserves_raw_input(self):
        result = classify_by_rules("  /todo buy milk  ")
        assert result.raw_input == "  /todo buy milk  ", "should preserve original input"

    def test_source_is_rules(self):
        result = classify_by_rules("hello world")
        assert result.source == "rules", "source should be 'rules'"

    @pytest.mark.parametrize("text", [
        "/TODO buy milk",
        "/Todo List",
        "REMIND me to buy milk",
        "?WHAT about deployment",
    ])
    def test_case_insensitive_matching(self, text):
        result = classify_by_rules(text)
        assert result.intent != Intent.CAPTURE, (
            f"'{text}' should not fall through to capture due to case"
        )


class TestStandaloneDatePatterns:
    @pytest.mark.parametrize("text", [
        "-1d",
        "-1w",
        "-2w",
        "-1m",
        "-3m",
        "-1y",
        "+1d",
        "+1w",
    ])
    def test_standalone_offsets_route_to_search(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.SEARCH, (
            f"standalone '{text}' should route to SEARCH"
        )
        assert result.confidence == Confidence.HIGH, (
            f"standalone '{text}' should have HIGH confidence"
        )

    @pytest.mark.parametrize("text", [
        "today",
        "yesterday",
        "tomorrow",
        "this week",
        "last week",
        "this month",
        "last month",
        "this year",
        "last year",
        "last monday",
        "last friday",
        "2026-03-28",
    ])
    def test_standalone_named_dates_route_to_search(self, text):
        result = classify_by_rules(text)
        assert result.intent == Intent.SEARCH, (
            f"standalone '{text}' should route to SEARCH"
        )
        assert result.confidence == Confidence.HIGH, (
            f"standalone '{text}' should have HIGH confidence"
        )

    @pytest.mark.parametrize("text", [
        "today I learned something new",
        "yesterday was a good day",
        "last week we shipped the feature",
    ])
    def test_date_with_extra_text_does_not_match_standalone(self, text):
        result = classify_by_rules(text)
        assert result.intent != Intent.SEARCH or result.confidence != Confidence.HIGH, (
            f"'{text}' should NOT match standalone date pattern"
        )
