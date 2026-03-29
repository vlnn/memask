from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(
        self,
        model_path: str | None,
        *,
        n_ctx: int = 2048,
        max_tokens: int = 512,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        self._model_path = model_path
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._model = None

    def is_available(self) -> bool:
        if self._model_path is None:
            return False
        return Path(self._model_path).is_file()

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        if not self.is_available():
            raise RuntimeError("LLM not available: model file missing or not configured")

        model = self._load_model()
        messages = self._build_messages(prompt, system)
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return self._extract_text(response)

    def _build_messages(self, prompt: str, system: str | None) -> list[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _extract_text(self, response: dict) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    def _load_model(self):
        if self._model is not None:
            return self._model
        from llama_cpp import Llama
        logger.info("loading LLM model: %s", self._model_path)
        self._model = Llama(
            model_path=self._model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            verbose=self._verbose,
        )
        return self._model
