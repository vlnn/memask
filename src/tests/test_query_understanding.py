from datetime import datetime

import pytest

from memask.router.query_understanding import extract_query_context


NOW = datetime(2026, 3, 30, 14, 30, 0)


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
    def test_strips_prefix_markers(self):
        ctx = extract_query_context("/todo list pending")
        assert "todo list pending" in ctx.raw_query, (
            "should strip slash prefix from raw_query"
        )


class TestSearchPreambleStripping:
    @pytest.mark.parametrize("text,expected_query", [
        ("what did I write about running?", "running"),
        ("what did I note about deployment?", "deployment"),
        ("what did I save about kubernetes?", "kubernetes"),
        ("what did I record about the meeting?", "meeting"),
        ("what did I jot down about testing?", "testing"),
        ("what were my notes about deployment?", "deployment"),
        ("what was my note about python?", "python"),
        ("what about deployment?", "deployment"),
        ("what was that thing about databases?", "databases"),
        ("what was that stuff about config?", "config"),
        ("find my notes about python", "python"),
        ("find notes about kubernetes", "kubernetes"),
        ("find deployment", "deployment"),
        ("search for python stuff", "python stuff"),
        ("look up meeting notes", "meeting notes"),
        ("look for deployment info", "deployment info"),
        ("show me notes about kubernetes", "kubernetes"),
        ("show me deployment stuff", "deployment stuff"),
        ("notes about python", "python"),
        ("my notes about the release plan", "release plan"),
        ("anything about docker?", "docker"),
        ("?what did I write about running?", "running"),
        ("?running", "running"),
    ])
    def test_strips_preamble_from_raw_query(self, text, expected_query):
        ctx = extract_query_context(text)
        assert ctx.raw_query == expected_query, (
            f"'{text}' should have raw_query '{expected_query}'"
        )

    @pytest.mark.parametrize("text,expected_query", [
        ("kubernetes cluster needs more RAM", "kubernetes cluster needs more RAM"),
        ("the meeting went well today", "meeting went well"),
        ("python 3.12 has nice new features", "python 3.12 has nice new features"),
    ])
    def test_preserves_non_search_text(self, text, expected_query):
        ctx = extract_query_context(text, now=NOW)
        assert ctx.raw_query == expected_query, (
            f"'{text}' should preserve raw text as '{expected_query}'"
        )


class TestDateRangeResolution:
    @pytest.mark.parametrize("text", [
        "notes from yesterday",
        "stuff from last week",
        "? things today",
        "/list -1d",
    ])
    def test_resolves_date_expressions(self, text):
        ctx = extract_query_context(text, now=NOW)
        assert ctx.date_range is not None, (
            f"'{text}' should resolve to a date range"
        )

    def test_no_date_range_for_plain_text(self):
        ctx = extract_query_context("kubernetes needs more RAM", now=NOW)
        assert ctx.date_range is None, (
            "plain text without date expression should have no date_range"
        )

    def test_yesterday_resolves_correctly(self):
        ctx = extract_query_context("notes from yesterday", now=NOW)
        assert ctx.date_range.label == "yesterday", "should resolve to 'yesterday'"

    def test_offset_resolves_correctly(self):
        ctx = extract_query_context("? deploy -1w", now=NOW)
        assert ctx.date_range.label == "last 1w", "should resolve -1w to 'last 1w'"

    def test_named_period_resolves(self):
        ctx = extract_query_context("? stuff from last week", now=NOW)
        assert ctx.date_range.label == "last week", (
            "should resolve 'last week'"
        )

    def test_iso_date_resolves(self):
        ctx = extract_query_context("? notes 2026-03-15", now=NOW)
        assert ctx.date_range.label == "2026-03-15", (
            "should resolve ISO date"
        )


class TestDateTokenStripping:
    @pytest.mark.parametrize("text,expected_query", [
        ("? deploy -1w", "deploy"),
        ("? notes from yesterday", "notes"),
        ("? deploy since last week", "deploy"),
        ("? stuff from last monday", "stuff"),
        ("-1d", ""),
        ("today", ""),
        ("last week", ""),
        ("/list -1w", "list"),
        ("notes from 2026-03-15", "notes"),
    ])
    def test_strips_date_tokens_from_raw_query(self, text, expected_query):
        ctx = extract_query_context(text, now=NOW)
        assert ctx.raw_query == expected_query, (
            f"'{text}' should have raw_query '{expected_query}' after stripping"
        )

    def test_strips_trailing_preposition(self):
        ctx = extract_query_context("notes from yesterday", now=NOW)
        assert "from" not in ctx.raw_query, (
            "trailing preposition 'from' should be stripped with the date"
        )

    def test_preserves_non_date_content(self):
        ctx = extract_query_context("? deploy to production -1w", now=NOW)
        assert "deploy to production" == ctx.raw_query, (
            "should preserve non-date content"
        )


class TestCombinedPreambleAndDate:
    @pytest.mark.parametrize("text,expected_query", [
        ("what did I write about running yesterday?", "running"),
        ("what did I note about deployment last week?", "deployment"),
        ("find my notes about kubernetes -1w", "kubernetes"),
        ("show me notes about docker from last monday", "docker"),
    ])
    def test_strips_both_preamble_and_date(self, text, expected_query):
        ctx = extract_query_context(text, now=NOW)
        assert ctx.raw_query == expected_query, (
            f"'{text}' should have raw_query '{expected_query}'"
        )
        assert ctx.date_range is not None, (
            f"'{text}' should also resolve a date range"
        )
