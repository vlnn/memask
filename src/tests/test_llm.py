import pytest

from memask.context import LLM
from memask.rag.llm import LocalLLM
from tests.helpers import FakeLLM


class TestFakeLLM:
    def test_generates_canned_response(self):
        llm = FakeLLM(response="42")
        assert llm.generate("what is the answer?") == "42", (
            "should return the canned response"
        )

    def test_is_available(self):
        llm = FakeLLM()
        assert llm.is_available() is True, "FakeLLM should always be available"

    def test_unavailable_when_configured(self):
        llm = FakeLLM(available=False)
        assert llm.is_available() is False, (
            "FakeLLM should report unavailable when configured"
        )

    def test_records_calls(self):
        llm = FakeLLM(response="ok")
        llm.generate("first")
        llm.generate("second", system="sys")
        assert len(llm.call_log) == 2, "should record all calls"
        assert llm.call_log[0] == {"prompt": "first", "system": None}, (
            "should record prompt and system"
        )
        assert llm.call_log[1] == {"prompt": "second", "system": "sys"}, (
            "should record system when provided"
        )

    def test_default_response(self):
        llm = FakeLLM()
        result = llm.generate("anything")
        assert isinstance(result, str), "should return a string"
        assert len(result) > 0, "default response should not be empty"

    def test_satisfies_llm_protocol(self):
        llm = FakeLLM()
        assert isinstance(llm, LLM), "FakeLLM should satisfy the LLM protocol"

    def test_sequential_responses(self):
        llm = FakeLLM(responses=["first", "second", "third"])
        assert llm.generate("a") == "first", "should return responses in order"
        assert llm.generate("b") == "second", "should advance to next response"
        assert llm.generate("c") == "third", "should return third response"

    def test_sequential_responses_cycle(self):
        llm = FakeLLM(responses=["only"])
        llm.generate("a")
        result = llm.generate("b")
        assert result == "only", "should repeat last response when exhausted"


class TestLocalLLMUnavailable:
    def test_unavailable_when_no_model_path(self):
        llm = LocalLLM(model_path=None)
        assert llm.is_available() is False, (
            "should be unavailable when model_path is None"
        )

    def test_unavailable_when_model_file_missing(self, tmp_path):
        fake_path = tmp_path / "nonexistent.gguf"
        llm = LocalLLM(model_path=str(fake_path))
        assert llm.is_available() is False, (
            "should be unavailable when model file does not exist"
        )

    def test_generate_raises_when_unavailable(self):
        llm = LocalLLM(model_path=None)
        with pytest.raises(RuntimeError, match="not available"):
            llm.generate("hello")

    def test_satisfies_llm_protocol(self):
        llm = LocalLLM(model_path=None)
        assert isinstance(llm, LLM), "LocalLLM should satisfy the LLM protocol"


class TestLocalLLMConfig:
    def test_default_context_size(self):
        llm = LocalLLM(model_path=None)
        assert llm.n_ctx == 2048, "default context size should be 2048"

    def test_custom_context_size(self):
        llm = LocalLLM(model_path=None, n_ctx=4096)
        assert llm.n_ctx == 4096, "should accept custom context size"

    def test_default_max_tokens(self):
        llm = LocalLLM(model_path=None)
        assert llm.max_tokens == 512, "default max_tokens should be 512"

    def test_custom_max_tokens(self):
        llm = LocalLLM(model_path=None, max_tokens=1024)
        assert llm.max_tokens == 1024, "should accept custom max_tokens"
