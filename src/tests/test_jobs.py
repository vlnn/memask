import json

from memask.repository.jobs import (
    dequeue,
    enqueue,
    get_job,
    mark_complete,
    mark_failed,
    queue_status,
    recover_stalled,
    requeue_if_retriable,
)


class TestEnqueue:
    def test_none_payload_by_default(self, conn):
        job = enqueue(conn, "embed")
        assert job.payload is None, "should store None when no payload given"

    def test_creates_pending_job(self, conn):
        job = enqueue(conn, "embed")
        assert job.status == "pending", "new job should be pending"
        assert job.type == "embed", "should store job type"
        assert job.attempts == 0, "should start with zero attempts"

    def test_stores_payload(self, conn):
        payload = {"item_id": "abc123"}
        job = enqueue(conn, "embed", payload=payload)
        assert job.payload is not None, "payload should be stored"
        assert json.loads(job.payload) == payload, "should serialize payload as JSON"

    def test_respects_max_attempts(self, conn):
        job = enqueue(conn, "embed", max_attempts=5)
        assert job.max_attempts == 5, "should store custom max_attempts"

    def test_default_max_attempts(self, conn):
        job = enqueue(conn, "embed")
        assert job.max_attempts == 3, "should default to 3 max attempts"


class TestDequeue:
    def test_returns_oldest_pending(self, conn):
        first = enqueue(conn, "embed")
        enqueue(conn, "embed")

        job = dequeue(conn)
        assert job is not None, "Job never is none"
        assert job.id == first.id, "should dequeue oldest pending job"
        assert job.status == "processing", "dequeued job should be processing"
        assert job.attempts == 1, "should increment attempts"

    def test_returns_none_when_empty(self, conn):
        assert dequeue(conn) is None, "should return None when queue is empty"

    def test_filters_by_type(self, conn):
        enqueue(conn, "embed")
        tag_job = enqueue(conn, "tag")

        job = dequeue(conn, job_type="tag")

        assert job is not None, "Job never is none"
        assert job.id == tag_job.id, "should dequeue only matching type"

    def test_skips_processing_jobs(self, conn):
        enqueue(conn, "embed")
        dequeue(conn)

        assert dequeue(conn) is None, "should not dequeue already-processing job"


class TestMarkComplete:
    def test_sets_complete_status(self, conn):
        enqueue(conn, "embed")
        processing = dequeue(conn)

        assert processing is not None, "Processing is always something"
        completed = mark_complete(conn, processing.id)
        assert completed is not None, "Complete status is never empty"
        assert completed.status == "complete", "should mark as complete"

    def test_completed_not_dequeued_again(self, conn):
        enqueue(conn, "embed")
        processing = dequeue(conn)
        mark_complete(conn, processing.id)

        assert dequeue(conn) is None, "completed job should not be dequeued"


class TestMarkFailed:
    def test_sets_failed_status_and_error(self, conn):
        enqueue(conn, "embed")
        processing = dequeue(conn)

        failed = mark_failed(conn, processing.id, "connection timeout")
        assert failed.status == "failed", "should mark as failed"
        assert failed.last_error == "connection timeout", "should store error message"


class TestRequeueIfRetriable:
    def test_requeues_when_under_max(self, conn):
        enqueue(conn, "embed", max_attempts=3)
        processing = dequeue(conn)

        requeued = requeue_if_retriable(conn, processing.id, "transient error")
        assert requeued.status == "pending", "should requeue retriable job"
        assert requeued.last_error == "transient error", "should store the error"

    def test_fails_when_at_max(self, conn):
        enqueue(conn, "embed", max_attempts=1)
        processing = dequeue(conn)

        result = requeue_if_retriable(conn, processing.id, "final error")
        assert result.status == "failed", "should fail when max attempts reached"

    def test_returns_none_for_missing(self, conn):
        result = requeue_if_retriable(conn, "ghost", "err")
        assert result is None, "should return None for missing job"


class TestRecoverStalled:
    def test_recovers_retriable_stalled_jobs(self, conn):
        enqueue(conn, "embed", max_attempts=3)
        job = dequeue(conn)

        recovered = recover_stalled(conn)
        assert recovered == 1, "should recover one stalled job"

        refreshed = get_job(conn, job.id)
        assert refreshed.status == "pending", "recovered job should be pending again"

    def test_fails_exhausted_stalled_jobs(self, conn):
        enqueue(conn, "embed", max_attempts=1)
        job = dequeue(conn)

        recover_stalled(conn)
        refreshed = get_job(conn, job.id)
        assert refreshed.status == "failed", "exhausted stalled job should be failed"
        assert "stalled" in refreshed.last_error, "should record stall reason"

    def test_no_op_when_nothing_stalled(self, conn):
        enqueue(conn, "embed")
        assert recover_stalled(conn) == 0, "should return 0 when nothing is stalled"

    def test_crash_recovery_scenario(self, conn):
        enqueue(conn, "embed", max_attempts=3)
        enqueue(conn, "tag", max_attempts=3)
        dequeue(conn)
        dequeue(conn)

        recover_stalled(conn)

        first = dequeue(conn)
        second = dequeue(conn)
        assert first is not None, "first recovered job should be dequeueable"
        assert second is not None, "second recovered job should be dequeueable"
        assert first.attempts == 2, "recovered job should have incremented attempts"


class TestQueueStatus:
    def test_empty_queue(self, conn):
        assert queue_status(conn) == {}, "empty queue should return empty dict"

    def test_counts_by_status(self, conn):
        enqueue(conn, "embed")
        enqueue(conn, "embed")
        enqueue(conn, "tag")
        dequeue(conn)

        status = queue_status(conn)
        assert status["pending"] == 2, "should count pending jobs"
        assert status["processing"] == 1, "should count processing jobs"
