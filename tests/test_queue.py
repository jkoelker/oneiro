"""Tests for GenerationQueue."""

import asyncio
from unittest.mock import AsyncMock, Mock

from oneiro.queue import GenerationQueue, QueueRequest, QueueResult, QueueStatus


class TestQueueStatus:
    """Tests for QueueStatus enum."""

    def test_queued_value(self):
        """QueueStatus.QUEUED has correct value."""
        assert QueueStatus.QUEUED.value == "queued"

    def test_user_limit_value(self):
        """QueueStatus.USER_LIMIT has correct value."""
        assert QueueStatus.USER_LIMIT.value == "user_limit"

    def test_global_limit_value(self):
        """QueueStatus.GLOBAL_LIMIT has correct value."""
        assert QueueStatus.GLOBAL_LIMIT.value == "global_limit"


class TestQueueResult:
    """Tests for QueueResult dataclass."""

    def test_default_values(self):
        """QueueResult has correct defaults."""
        result = QueueResult(status=QueueStatus.QUEUED)
        assert result.status == QueueStatus.QUEUED
        assert result.position == 0
        assert result.message == ""

    def test_custom_values(self):
        """QueueResult accepts custom values."""
        result = QueueResult(
            status=QueueStatus.USER_LIMIT,
            position=5,
            message="Limit reached",
        )
        assert result.status == QueueStatus.USER_LIMIT
        assert result.position == 5
        assert result.message == "Limit reached"


class TestQueueRequest:
    """Tests for QueueRequest dataclass."""

    def test_auto_generates_id(self):
        """QueueRequest generates a unique ID."""
        callback = AsyncMock()
        request = QueueRequest(user_id=123, request={}, callback=callback)
        assert request.id != ""
        assert len(request.id) == 8

    def test_unique_ids(self):
        """Each QueueRequest gets a unique ID."""
        callback = AsyncMock()
        request1 = QueueRequest(user_id=123, request={}, callback=callback)
        request2 = QueueRequest(user_id=123, request={}, callback=callback)
        assert request1.id != request2.id

    def test_custom_id(self):
        """QueueRequest can use custom ID."""
        callback = AsyncMock()
        request = QueueRequest(user_id=123, request={}, callback=callback, id="custom")
        assert request.id == "custom"

    def test_optional_callbacks(self):
        """QueueRequest optional callbacks default to None."""
        callback = AsyncMock()
        request = QueueRequest(user_id=123, request={}, callback=callback)
        assert request.on_start is None
        assert request.on_position_update is None


class TestGenerationQueueInit:
    """Tests for GenerationQueue initialization."""

    def test_default_limits(self):
        """GenerationQueue has correct default limits."""
        queue = GenerationQueue()
        assert queue.max_global == 100
        assert queue.max_per_user == 20

    def test_custom_limits(self):
        """GenerationQueue accepts custom limits."""
        queue = GenerationQueue(max_global=50, max_per_user=5)
        assert queue.max_global == 50
        assert queue.max_per_user == 5

    def test_initial_state(self):
        """GenerationQueue starts empty."""
        queue = GenerationQueue()
        assert queue.size == 0
        assert queue._running is False
        assert queue._worker_task is None


class TestGenerationQueueSize:
    """Tests for GenerationQueue.size property."""

    def test_size_empty(self):
        """Size is 0 for empty queue."""
        queue = GenerationQueue()
        assert queue.size == 0

    def test_size_after_add(self):
        """Size increases after adding requests."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        assert queue.size == 1
        queue.add(user_id=2, request={}, callback=callback)
        assert queue.size == 2


class TestGenerationQueueUserCount:
    """Tests for GenerationQueue.user_count()."""

    def test_user_count_empty(self):
        """User count is 0 for unknown user."""
        queue = GenerationQueue()
        assert queue.user_count(12345) == 0

    def test_user_count_after_add(self):
        """User count increases after adding requests."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=123, request={}, callback=callback)
        assert queue.user_count(123) == 1
        queue.add(user_id=123, request={}, callback=callback)
        assert queue.user_count(123) == 2

    def test_user_count_per_user(self):
        """User counts are tracked separately per user."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        queue.add(user_id=1, request={}, callback=callback)
        queue.add(user_id=2, request={}, callback=callback)
        assert queue.user_count(1) == 2
        assert queue.user_count(2) == 1


class TestGenerationQueueAdd:
    """Tests for GenerationQueue.add()."""

    def test_add_returns_queued(self):
        """Add returns QUEUED status for valid request."""
        queue = GenerationQueue()
        callback = AsyncMock()
        result = queue.add(user_id=1, request={}, callback=callback)
        assert result.status == QueueStatus.QUEUED
        assert result.position == 1

    def test_add_position_increments(self):
        """Add returns incrementing positions."""
        queue = GenerationQueue()
        callback = AsyncMock()
        result1 = queue.add(user_id=1, request={}, callback=callback)
        result2 = queue.add(user_id=2, request={}, callback=callback)
        result3 = queue.add(user_id=3, request={}, callback=callback)
        assert result1.position == 1
        assert result2.position == 2
        assert result3.position == 3

    def test_add_rejects_at_global_limit(self):
        """Add rejects when global limit reached."""
        queue = GenerationQueue(max_global=2)
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        queue.add(user_id=2, request={}, callback=callback)
        result = queue.add(user_id=3, request={}, callback=callback)
        assert result.status == QueueStatus.GLOBAL_LIMIT
        assert "full" in result.message.lower()

    def test_add_rejects_at_user_limit(self):
        """Add rejects when user limit reached."""
        queue = GenerationQueue(max_per_user=2)
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        queue.add(user_id=1, request={}, callback=callback)
        result = queue.add(user_id=1, request={}, callback=callback)
        assert result.status == QueueStatus.USER_LIMIT
        assert "pending" in result.message.lower()

    def test_add_allows_other_user_at_user_limit(self):
        """Add allows different user even when one user is at limit."""
        queue = GenerationQueue(max_per_user=1)
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        result = queue.add(user_id=2, request={}, callback=callback)
        assert result.status == QueueStatus.QUEUED

    def test_add_processing_message_for_first(self):
        """Add returns 'Processing...' message for first request."""
        queue = GenerationQueue()
        callback = AsyncMock()
        result = queue.add(user_id=1, request={}, callback=callback)
        assert "Processing" in result.message

    def test_add_position_message_for_later(self):
        """Add returns position message for queued requests."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        result = queue.add(user_id=2, request={}, callback=callback)
        assert "position 2" in result.message


class TestGenerationQueueGetUserPosition:
    """Tests for GenerationQueue._get_user_position()."""

    def test_get_user_position_empty(self):
        """Returns 0 for user with no requests."""
        queue = GenerationQueue()
        assert queue._get_user_position(123) == 0

    def test_get_user_position_first(self):
        """Returns 1 for user's first request at front."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=123, request={}, callback=callback)
        assert queue._get_user_position(123) == 1

    def test_get_user_position_later(self):
        """Returns correct position when behind others."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=1, request={}, callback=callback)
        queue.add(user_id=2, request={}, callback=callback)
        queue.add(user_id=123, request={}, callback=callback)
        assert queue._get_user_position(123) == 3


class TestGenerationQueueRemoveRequest:
    """Tests for GenerationQueue._remove_request()."""

    def test_remove_request_decrements_count(self):
        """Remove decrements user count."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=123, request={}, callback=callback)
        assert queue.user_count(123) == 1
        request = queue._pending_requests[0]
        queue._remove_request(request)
        assert queue.user_count(123) == 0

    def test_remove_request_decrements_size(self):
        """Remove decrements queue size."""
        queue = GenerationQueue()
        callback = AsyncMock()
        queue.add(user_id=123, request={}, callback=callback)
        assert queue.size == 1
        request = queue._pending_requests[0]
        queue._remove_request(request)
        assert queue.size == 0

    def test_remove_request_handles_unknown(self):
        """Remove handles unknown request gracefully."""
        queue = GenerationQueue()
        callback = AsyncMock()
        request = QueueRequest(user_id=123, request={}, callback=callback)
        # Should not raise
        queue._remove_request(request)


class TestGenerationQueueStartStop:
    """Tests for GenerationQueue.start() and stop()."""

    async def test_start_sets_running(self):
        """Start sets running flag."""
        queue = GenerationQueue()
        pipeline = Mock()
        await queue.start(pipeline)
        assert queue._running is True
        await queue.stop()

    async def test_start_creates_worker(self):
        """Start creates worker task."""
        queue = GenerationQueue()
        pipeline = Mock()
        await queue.start(pipeline)
        assert queue._worker_task is not None
        await queue.stop()

    async def test_start_idempotent(self):
        """Start is idempotent."""
        queue = GenerationQueue()
        pipeline = Mock()
        await queue.start(pipeline)
        task1 = queue._worker_task
        await queue.start(pipeline)
        task2 = queue._worker_task
        assert task1 is task2
        await queue.stop()

    async def test_stop_clears_running(self):
        """Stop clears running flag."""
        queue = GenerationQueue()
        pipeline = Mock()
        await queue.start(pipeline)
        await queue.stop()
        assert queue._running is False

    async def test_stop_clears_worker(self):
        """Stop clears worker task."""
        queue = GenerationQueue()
        pipeline = Mock()
        await queue.start(pipeline)
        await queue.stop()
        assert queue._worker_task is None


class TestGenerationQueueProcessing:
    """Tests for queue processing behavior."""

    async def test_processes_request(self):
        """Queue processes request and calls callback."""
        queue = GenerationQueue()
        pipeline = Mock()
        pipeline.generate = AsyncMock(return_value="result")
        callback = AsyncMock()

        await queue.start(pipeline)
        queue.add(user_id=1, request={"prompt": "test"}, callback=callback)

        # Give worker time to process
        await asyncio.sleep(0.1)
        await queue.stop()

        pipeline.generate.assert_called_once_with(prompt="test")
        callback.assert_called_once_with("result")

    async def test_calls_on_start(self):
        """Queue calls on_start callback before processing."""
        queue = GenerationQueue()
        pipeline = Mock()
        pipeline.generate = AsyncMock(return_value="result")
        callback = AsyncMock()
        on_start = AsyncMock()

        await queue.start(pipeline)
        queue.add(user_id=1, request={}, callback=callback, on_start=on_start)

        await asyncio.sleep(0.1)
        await queue.stop()

        on_start.assert_called_once()

    async def test_handles_generation_error(self):
        """Queue passes exception to callback on error."""
        queue = GenerationQueue()
        pipeline = Mock()
        error = ValueError("Generation failed")
        pipeline.generate = AsyncMock(side_effect=error)
        callback = AsyncMock()

        await queue.start(pipeline)
        queue.add(user_id=1, request={}, callback=callback)

        await asyncio.sleep(0.1)
        await queue.stop()

        callback.assert_called_once()
        called_arg = callback.call_args[0][0]
        assert isinstance(called_arg, ValueError)
        assert "Generation failed" in str(called_arg)
