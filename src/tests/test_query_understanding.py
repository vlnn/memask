import pytest

from memask.router.query_understanding import extract_query_context


class TestDateHintExtraction:
    @pytest.mark.parametrize("text,expected_hints", [
        ("notes from yesterday", ["yesterday"]),
        ("what did I write today", ["today"]),
        ("notes from last week", ["last week"]),
        ("stuff from this morning", ["this morning"]),
        ("things I wrote last monday", ["last monday"]),
        ("meeting notes from march", ["march"]),
        ("no date here", []),
        ("deployed yesterday and today", ["yesterday", "today"]),
    ])
    def test_extracts_date_hints(self, text, expected_hints):
        ctx = extract_query_context(text)
        assert ctx.date_hints == expected_hints, (
            f"'{text}' should extract date hints {expected_hints}"
        )


class TestTopicExtraction:
    @pytest.mark.parametrize("text,expected_topic", [
        ("what did I note about deployment", "deployment"),
        ("find notes about kubernetes", "kubernetes"),
        ("search for python stuff", "python stuff"),
        ("notes on the release plan", "release plan"),
        ("what about the database migration", "database migration"),
    ])
    def test_extracts_topic_from_about_phrases(self, text, expected_topic):
        ctx = extract_query_context(text)
        assert ctx.topic == expected_topic, (
            f"'{text}' should extract topic '{expected_topic}'"
        )

    def test_no_topic_for_plain_text(self):
        ctx = extract_query_context("the sky is blue")
        assert ctx.topic is None, "plain text should have no topic"


class TestTypeFilterExtraction:
    @pytest.mark.parametrize("text,expected_type", [
        ("show my todos", "todo"),
        ("find my todo items", "todo"),
        ("list all notes", "note"),
        ("search my notes about python", "note"),
        ("what decisions did I make", "decision"),
        ("random text here", None),
    ])
    def test_extracts_type_filter(self, text, expected_type):
        ctx = extract_query_context(text)
        assert ctx.type_filter == expected_type, (
            f"'{text}' should extract type filter '{expected_type}'"
        )


class TestStatusFilterExtraction:
    @pytest.mark.parametrize("text,expected_status", [
        ("show pending todos", "pending"),
        ("list done tasks", "done"),
        ("completed items", "done"),
        ("open tasks", "pending"),
        ("random text", None),
    ])
    def test_extracts_status_filter(self, text, expected_status):
        ctx = extract_query_context(text)
        assert ctx.status_filter == expected_status, (
            f"'{text}' should extract status filter '{expected_status}'"
        )


class TestRawQuery:
    def test_stores_cleaned_query(self):
        ctx = extract_query_context("?what about deployment yesterday")
        assert ctx.raw_query == "what about deployment yesterday", (
            "should strip prefix markers from raw_query"
        )

    def test_strips_prefix_markers(self):
        ctx = extract_query_context("/todo list pending")
        assert ctx.raw_query == "todo list pending", (
            "should strip slash prefix from raw_query"
        )
