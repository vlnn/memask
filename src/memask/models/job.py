import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    id: str
    type: str
    status: str
    payload: str | None
    attempts: int
    max_attempts: int
    last_error: str | None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(**dict(row))
