import sqlite3
from uuid import uuid4

from memask.clock import now


def create_instruction(conn: sqlite3.Connection, content: str) -> dict:
    instruction_id = str(uuid4())
    timestamp = now()
    conn.execute(
        "INSERT INTO instructions (id, content, active, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
        (instruction_id, content, timestamp, timestamp),
    )
    conn.commit()
    return {
        "id": instruction_id,
        "content": content,
        "active": 1,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def list_active_instructions(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, content, active, created_at, updated_at FROM instructions WHERE active = 1 ORDER BY created_at",
    ).fetchall()
    return [
        {
            "id": row[0],
            "content": row[1],
            "active": row[2],
            "created_at": row[3],
            "updated_at": row[4],
        }
        for row in rows
    ]


def deactivate_instruction(conn: sqlite3.Connection, instruction_id: str) -> bool:
    cursor = conn.execute(
        "UPDATE instructions SET active = 0, updated_at = ? WHERE id = ? AND active = 1",
        (now(), instruction_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def deactivate_all_instructions(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "UPDATE instructions SET active = 0, updated_at = ? WHERE active = 1",
        (now(),),
    )
    conn.commit()
    return cursor.rowcount
