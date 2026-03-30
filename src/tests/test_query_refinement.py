import pytest

from memask.router.query_refinement import (
    needs_refinement,
    refine_search_query,
    _clean_llm_response,
)
from tests.helpers import FakeLLM


class TestNeedsRefinement:
    @pytest.mark.parametrize("query", [
        "what did I write about running",
        "where did I put the config notes",
        "how did I set up the database",
        "when was the last deployment done",
        "who suggested the new architecture",
        "which framework did I choose for this",
    ])
    def test_natural_language_queries_need_refinement(self, query):
        assert needs_refinement(query) is True, (
            f"'{query}' should need refinement"
        )

    @pytest.mark.parametrize("query", [
        "running",
        "deployment",
        "kubernetes config",
        "deploy production",
        "",
        "   ",
    ])
    def test_keyword_queries_skip_refinement(self, query):
        assert needs_refinement(query) is False, (
            f"'{query}' should not need refinement"
        )

    @pytest.mark.parametrize("query", [
        "is it done",
        "did we do it",
    ])
    def test_short_questions_skip_refinement(self, query):
        assert needs_refinement(query) is False, (
            f"'{query}' is too short to refine"
        )

    def test_no_question_words_skips(self):
        assert needs_refinement("deploy to production environment") is False, (
            "no question words means no refinement needed"
        )


class TestRefineSearchQuery:
    def test_uses_llm_to_extract_keywords(self):
        llm = FakeLLM(response="running")
        result = refine_search_query("what did I write about running", llm)
        assert result == "running", "should return LLM-extracted keywords"

    def test_passes_query_in_prompt(self):
        llm = FakeLLM(response="deployment")
        refine_search_query("where are my deployment notes", llm)
        assert "where are my deployment notes" in llm.call_log[0]["prompt"], (
            "should include original query in prompt"
        )

    def test_passes_system_message(self):
        llm = FakeLLM(response="keywords")
        refine_search_query("some query here please", llm)
        assert llm.call_log[0]["system"] is not None, (
            "should send a system prompt"
        )
        assert "keyword" in llm.call_log[0]["system"].lower(), (
            "system prompt should mention keyword extraction"
        )

    def test_falls_back_on_llm_exception(self):
        class BrokenLLM:
            def generate(self, prompt, *, system=None):
                raise RuntimeError("model crashed")
            def is_available(self):
                return True

        result = refine_search_query("what about deployment", BrokenLLM())
        assert result == "what about deployment", (
            "should return original query when LLM fails"
        )

    def test_falls_back_on_empty_response(self):
        llm = FakeLLM(response="   ")
        result = refine_search_query("what about deployment", llm)
        assert result == "what about deployment", (
            "should return original query when LLM returns empty"
        )

    @pytest.mark.parametrize("query,llm_response,expected", [
        ("what did I write about running", "running", "running"),
        ("how did I configure postgres", "configure postgres", "configure postgres"),
        ("where are the deployment notes", "deployment notes", "deployment notes"),
    ])
    def test_extracts_keywords_for_various_queries(self, query, llm_response, expected):
        llm = FakeLLM(response=llm_response)
        result = refine_search_query(query, llm)
        assert result == expected, (
            f"'{query}' with LLM response '{llm_response}' should produce '{expected}'"
        )


class TestCleanLlmResponse:
    @pytest.mark.parametrize("response,expected", [
        ("running", "running"),
        ("  running  ", "running"),
        ('"running"', "running"),
        ("'running'", "running"),
        ("Keywords: running", "running"),
        ("keyword: running", "running"),
        ("Search terms: deployment config", "deployment config"),
        ("search for: deployment", "deployment"),
        ("running.", "running"),
        ("running!", "running"),
        ("running?\n\nExtra stuff", "running"),
    ])
    def test_cleans_various_formats(self, response, expected):
        result = _clean_llm_response(response)
        assert result == expected, (
            f"'{response}' should clean to '{expected}'"
        )


class TestRefinementIntegration:
    def test_short_query_unchanged(self):
        llm = FakeLLM(response="should not be called")
        query = "deployment"
        assert not needs_refinement(query), "short query should skip refinement"

    def test_long_natural_query_refined(self):
        llm = FakeLLM(response="running VPN")
        query = "what did I write about running VPN"
        assert needs_refinement(query), "natural language query should need refinement"
        result = refine_search_query(query, llm)
        assert result == "running VPN", "should extract core keywords"
