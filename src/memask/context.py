from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_one(self, text: str) -> Any: ...

    def embed_many(self, texts: list[str]) -> Any: ...


@runtime_checkable
class LLM(Protocol):
    def generate(self, prompt: str, *, system: str | None = None) -> str: ...

    def is_available(self) -> bool: ...


@runtime_checkable
class Reranker(Protocol):
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass
class ServiceContext:
    conn: sqlite3.Connection
    store: Any | None = None
    embedder: Embedder | None = None
    llm: LLM | None = None
    reranker: Reranker | None = None
    session: Any | None = None
