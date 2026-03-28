from dataclasses import dataclass
from sqlite3 import Row


@dataclass(frozen=True)
class Item:
    id: str
    type: str
    content: str
    created_at: str
    updated_at: str
    title: str | None = None
    status: str | None = None
    priority: int | None = None
    due_date: str | None = None
    category: str | None = None
    source: str = "manual"
    tags: str | None = None
    deleted_at: str | None = None
    metadata: str | None = None

    @classmethod
    def from_row(cls, row: Row) -> "Item":
        return cls(**dict(row))
