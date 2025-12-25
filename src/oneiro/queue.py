"""Async generation queue with per-user limits."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oneiro.pipelines import PipelineManager


class QueueStatus(Enum):
    """Status of a queue add operation."""

    QUEUED = "queued"
    USER_LIMIT = "user_limit"
    GLOBAL_LIMIT = "global_limit"


@dataclass
class QueueResult:
    """Result of attempting to add to queue."""

    status: QueueStatus
    position: int = 0
    message: str = ""


@dataclass
class QueueRequest:
    """A request in the generation queue."""

    user_id: int
    request: dict[str, Any]
    callback: Callable[[Any], Coroutine[Any, Any, None]]
    on_start: Callable[[], Coroutine[Any, Any, None]] | None = None
    on_position_update: Callable[[int], Coroutine[Any, Any, None]] | None = None
    id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


class GenerationQueue:
    """Async queue for generation requests with per-user limits.

    Features:
    - Single worker task serializes GPU access
    - Per-user limit (default 20)
    - Global limit (default 100)
    - No expiration - all queued requests eventually processed
    """

    def __init__(self, max_global: int = 100, max_per_user: int = 20):
        self.max_global = max_global
        self.max_per_user = max_per_user

        self._queue: asyncio.Queue[QueueRequest] = asyncio.Queue()
        self._user_counts: dict[int, int] = defaultdict(int)
        self._pending_requests: list[QueueRequest] = []  # For position tracking
        self._worker_task: asyncio.Task | None = None
        self._pipeline: PipelineManager | None = None
        self._running = False

    @property
    def size(self) -> int:
        """Current number of requests in queue."""
        return len(self._pending_requests)

    def user_count(self, user_id: int) -> int:
        """Number of pending requests for a user."""
        return self._user_counts[user_id]

    async def start(self, pipeline_manager: "PipelineManager") -> None:
        """Start the queue worker."""
        if self._running:
            return

        self._pipeline = pipeline_manager
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        print("Queue worker started")

    async def stop(self) -> None:
        """Stop the queue worker gracefully."""
        self._running = False
        if self._worker_task:
            # Put a sentinel to unblock the worker
            await self._queue.put(None)  # type: ignore
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except TimeoutError:
                self._worker_task.cancel()
            self._worker_task = None
        print("Queue worker stopped")

    def add(
        self,
        user_id: int,
        request: dict[str, Any],
        callback: Callable,
        on_start: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_position_update: Callable[[int], Coroutine[Any, Any, None]] | None = None,
    ) -> QueueResult:
        """Add a request to the queue.

        Args:
            user_id: Discord user ID
            request: Generation parameters dict
            callback: Async function to call with result (or exception)
            on_start: Async function called when generation starts
            on_position_update: Async function called with new position when queue changes

        Returns:
            QueueResult with status and position
        """
        # Check global limit
        if self.size >= self.max_global:
            return QueueResult(
                status=QueueStatus.GLOBAL_LIMIT,
                position=self.size,
                message=f"Queue is full ({self.max_global} requests). Please try again later.",
            )

        # Check per-user limit
        if self._user_counts[user_id] >= self.max_per_user:
            return QueueResult(
                status=QueueStatus.USER_LIMIT,
                position=self._get_user_position(user_id),
                message=f"You have {self.max_per_user} requests pending. Please wait for some to complete.",
            )

        # Create and queue the request
        queue_request = QueueRequest(
            user_id=user_id,
            request=request,
            callback=callback,
            on_start=on_start,
            on_position_update=on_position_update,
        )

        self._user_counts[user_id] += 1
        self._pending_requests.append(queue_request)
        self._queue.put_nowait(queue_request)

        position = len(self._pending_requests)
        return QueueResult(
            status=QueueStatus.QUEUED,
            position=position,
            message=f"Queued at position {position}" if position > 1 else "Processing...",
        )

    def _get_user_position(self, user_id: int) -> int:
        """Get position of user's first request in queue."""
        for i, req in enumerate(self._pending_requests):
            if req.user_id == user_id:
                return i + 1
        return 0

    def _remove_request(self, request: QueueRequest) -> None:
        """Remove a completed request from tracking."""
        if request in self._pending_requests:
            self._pending_requests.remove(request)
        if self._user_counts[request.user_id] > 0:
            self._user_counts[request.user_id] -= 1

    async def _worker(self) -> None:
        """Worker task that processes queued requests sequentially."""
        print("Queue worker running...")

        while self._running:
            try:
                # Wait for next request
                request = await self._queue.get()

                # Check for stop sentinel
                if request is None:
                    break

                await self._process_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue worker error: {e}")
                # Continue processing other requests

        print("Queue worker exiting")

    async def _notify_position_updates(self) -> None:
        for i, req in enumerate(self._pending_requests):
            if req.on_position_update:
                try:
                    await req.on_position_update(i + 1)
                except Exception as e:
                    print(f"Position update callback error: {e}")

    async def _process_request(self, request: QueueRequest) -> None:
        if self._pipeline is None:
            print("No pipeline available, skipping request")
            self._remove_request(request)
            return

        try:
            print(f"Processing request {request.id} from user {request.user_id}")

            if request.on_start:
                try:
                    await request.on_start()
                except Exception as e:
                    print(f"on_start callback error: {e}")

            result = await self._pipeline.generate(**request.request)
            await request.callback(result)

        except Exception as e:
            print(f"Generation error for request {request.id}: {e}")
            try:
                await request.callback(e)
            except Exception as callback_error:
                print(f"Callback error: {callback_error}")

        finally:
            self._remove_request(request)
            self._queue.task_done()
            await self._notify_position_updates()
