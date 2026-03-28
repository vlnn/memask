import numpy as np
import pytest


class TestFakeEmbeddingService:
    def test_embed_one_returns_vector(self, fake_embedder):
        vec = fake_embedder.embed_one("hello")
        assert vec.shape == (384,), "should return vector of configured dimension"

    def test_embed_one_is_normalized(self, fake_embedder):
        vec = fake_embedder.embed_one("hello")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, "vector should be unit-normalized"

    def test_embed_one_is_deterministic(self, fake_embedder):
        a = fake_embedder.embed_one("same text")
        b = fake_embedder.embed_one("same text")
        assert np.allclose(a, b), "same text should produce same vector"

    def test_different_text_different_vector(self, fake_embedder):
        a = fake_embedder.embed_one("hello")
        b = fake_embedder.embed_one("world")
        assert not np.allclose(a, b), "different text should produce different vectors"

    def test_embed_many(self, fake_embedder):
        vecs = fake_embedder.embed_many(["hello", "world", "foo"])
        assert vecs.shape == (3, 384), "should return matrix with one row per text"

    def test_embed_many_empty(self, fake_embedder):
        vecs = fake_embedder.embed_many([])
        assert len(vecs) == 0, "empty input should return empty array"

    def test_logs_calls(self, fake_embedder):
        fake_embedder.embed_one("test")
        fake_embedder.embed_many(["a", "b"])
        assert len(fake_embedder.call_log) == 2, "should log all calls"
        assert fake_embedder.call_log[0] == ("embed_one", "test"), "should log embed_one"
        assert fake_embedder.call_log[1] == ("embed_many", ["a", "b"]), "should log embed_many"

    def test_model_name(self, fake_embedder):
        assert fake_embedder.model_name == "test-model", "should expose model name"
