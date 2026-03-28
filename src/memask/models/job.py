from dataclasses import dataclass
from sqlite3 import Row


@dataclass(frozen=True)
class Job:
    id: str
    type: str
    status: str
    attempts: int
    max_attempts: int
    created_at: str
    updated_at: str
    payload: str | None = None
    last_error: str | None = None

    @classmethod
    def from_row(cls, row: Row) -> "Job":
        return cls(**dict(row))
