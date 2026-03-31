import sqlite3
from dataclasses import dataclass

FIELDS = frozenset(
    {
        "id",
        "type",
        "content",
        "title",
        "status",
        "priority",
        "due_date",
        "category",
        "source",
        "tags",
        "created_at",
        "updated_at",
        "deleted_at",
        "metadata",
    }
)


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

    def to_summary(self) -> dict:
        return {"id": self.id, "content": self.content}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "title": self.title,
            "status": self.status,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_list_entry(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Item":
        d = dict(row)
        filtered = {k: v for k, v in d.items() if k in FIELDS}
        return cls(**filtered)
