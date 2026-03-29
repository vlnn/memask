import json
import sqlite3

from memask.clock import now
from memask.models.job import Job
from memask.ulid import ulid


def enqueue(
    conn: sqlite3.Connection,
    job_type: str,
    payload: dict | None = None,
    max_attempts: int = 3,
) -> Job:
    job_id = ulid()
    timestamp = now()
    payload_json = json.dumps(payload) if payload else None

    conn.execute(
        """INSERT INTO jobs
           (id, type, status, payload, attempts,
            max_attempts, created_at, updated_at)
           VALUES (?, ?, 'pending', ?, 0, ?, ?, ?)""",
        (job_id, job_type, payload_json, max_attempts, timestamp, timestamp),
    )
    conn.commit()
    return get_job(conn, job_id)


def get_job(conn: sqlite3.Connection, job_id: str) -> Job | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return Job.from_row(row) if row else None


def dequeue(conn: sqlite3.Connection, job_type: str | None = None) -> Job | None:
    conditions = ["status = 'pending'"]
    params: list = []
    if job_type:
        conditions.append("type = ?")
        params.append(job_type)

    where = " AND ".join(conditions)
    row = conn.execute(
        f"SELECT * FROM jobs WHERE {where} ORDER BY created_at ASC LIMIT 1",
        params,
    ).fetchone()

    if not row:
        return None

    job = Job.from_row(row)
    conn.execute(
        "UPDATE jobs SET status = 'processing',"
        " attempts = attempts + 1, updated_at = ?"
        " WHERE id = ?",
        (now(), job.id),
    )
    conn.commit()
    return get_job(conn, job.id)


def mark_complete(conn: sqlite3.Connection, job_id: str) -> Job | None:
    conn.execute(
        "UPDATE jobs SET status = 'complete', updated_at = ? WHERE id = ?",
        (now(), job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def mark_failed(
    conn: sqlite3.Connection,
    job_id: str,
    error: str,
) -> Job | None:
    conn.execute(
        "UPDATE jobs SET status = 'failed',"
        " last_error = ?, updated_at = ?"
        " WHERE id = ?",
        (error, now(), job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def requeue_if_retriable(
    conn: sqlite3.Connection,
    job_id: str,
    error: str,
) -> Job | None:
    job = get_job(conn, job_id)
    if not job:
        return None

    if job.attempts >= job.max_attempts:
        new_status = "failed"
    else:
        new_status = "pending"

    conn.execute(
        "UPDATE jobs SET status = ?, last_error = ?, updated_at = ? WHERE id = ?",
        (new_status, error, now(), job_id),
    )
    conn.commit()
    return get_job(conn, job_id)


def queue_status(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
    ).fetchall()
    return {row["status"]: row["count"] for row in rows}


def recover_stalled(conn: sqlite3.Connection) -> int:
    timestamp = now()
    conn.execute(
        "UPDATE jobs SET status = 'failed',"
        " last_error = 'stalled: exceeded max attempts',"
        " updated_at = ?"
        " WHERE status = 'processing'"
        " AND attempts >= max_attempts",
        (timestamp,),
    )
    cursor = conn.execute(
        "UPDATE jobs SET status = 'pending', updated_at = ?"
        " WHERE status = 'processing'",
        (timestamp,),
    )
    conn.commit()
    return cursor.rowcount
