from __future__ import annotations

from collections import deque

DEFAULT_MAX_EXCHANGES = 10


class Session:
    def __init__(self, *, max_exchanges: int = DEFAULT_MAX_EXCHANGES):
        self.max_exchanges = max_exchanges
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=max_exchanges)

    def add_exchange(self, query: str, answer: str) -> None:
        self._exchanges.append((query, answer))

    def history(self) -> list[dict[str, str]]:
        entries = []
        for query, answer in self._exchanges:
            entries.append({"role": "user", "content": query})
            entries.append({"role": "assistant", "content": answer})
        return [dict(e) for e in entries]

    def clear(self) -> None:
        self._exchanges.clear()

    def __len__(self) -> int:
        return len(self._exchanges)
