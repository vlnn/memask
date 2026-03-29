import numpy as np


class FakeEmbeddingService:
    def __init__(self, dimension=384, model_name="test-model"):
        self._model_name = model_name
        self._dimension = dimension
        self._call_log = []

    @property
    def model_name(self):
        return self._model_name

    @property
    def dimension(self):
        return self._dimension

    @property
    def call_log(self):
        return self._call_log

    def embed_one(self, text):
        self._call_log.append(("embed_one", text))
        return self._deterministic_vector(text)

    def embed_many(self, texts):
        self._call_log.append(("embed_many", texts))
        return np.array([self._deterministic_vector(t) for t in texts])

    def _deterministic_vector(self, text):
        rng = np.random.RandomState(hash(text) % 2**31)
        vec = rng.randn(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class FakeLLM:
    def __init__(
        self,
        response: str = "This is a test response.",
        available: bool = True,
        responses: list[str] | None = None,
    ):
        self._responses = responses or [response]
        self._call_index = 0
        self._available = available
        self.call_log: list[dict] = []

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.call_log.append({"prompt": prompt, "system": system})
        idx = min(self._call_index, len(self._responses) - 1)
        self._call_index += 1
        return self._responses[idx]

    def is_available(self) -> bool:
        return self._available


class FakeReranker:
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._word_overlap_score(a, b) for a, b in pairs]

    def _word_overlap_score(self, text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        total = len(words_a | words_b)
        return overlap / total if total > 0 else 0.0
