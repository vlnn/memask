import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def embed_one(self, text: str) -> np.ndarray:
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(texts, normalize_embeddings=True)

    def _load_model(self):
        if self._model is None:
            import logging
            import os

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            for name in ("sentence_transformers", "transformers", "huggingface_hub"):
                logging.getLogger(name).setLevel(logging.ERROR)

            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(
                    self._model_name, local_files_only=True
                )
            except OSError:
                self._model = SentenceTransformer(self._model_name)
        return self._model
