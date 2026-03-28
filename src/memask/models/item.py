from dataclasses import dataclass, fields
from sqlite3 import Row

_ITEM_FIELDS: set[str] = set()


@dataclass(frozen=True)
class Item:
    id: str
    type: str
    content: str
    title: str | None
    status: str | None
    priority: int | None
    due_date: str | None
    category: str | None
    source: str
    tags: str | None
    created_at: str
    updated_at: str
    deleted_at: str | None
    metadata: str | None

    @staticmethod
    def from_row(row: Row) -> "Item":
        global _ITEM_FIELDS
        if not _ITEM_FIELDS:
            _ITEM_FIELDS = {f.name for f in fields(Item)}
        data = {k: v for k, v in dict(row).items() if k in _ITEM_FIELDS}
        return Item(**data)
