import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_one(self, text: str) -> np.ndarray:
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True)
