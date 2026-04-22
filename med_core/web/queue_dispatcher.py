"""Training queue dispatchers (local and Redis-backed)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class LocalTrainingQueueDispatcher:
    """Direct dispatch: enqueue means schedule immediately in current process."""

    def __init__(self, *, on_job: Callable[[str], None]) -> None:
        self._on_job = on_job

    async def start(self) -> None:  # noqa: D401
        return None

    async def stop(self) -> None:  # noqa: D401
        return None

    def enqueue(self, job_id: str) -> None:
        self._on_job(job_id)


class RedisTrainingQueueDispatcher:
    """Redis list-backed dispatcher with one local consumer loop."""

    def __init__(
        self,
        *,
        redis_url: str,
        queue_name: str,
        on_job: Callable[[str], None],
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis dispatcher requires `redis` package. Install web extras first."
            ) from exc

        self._redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self._queue_name = queue_name
        self._on_job = on_job
        self._consumer_task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._consumer_task is not None and not self._consumer_task.done():
            return
        self._stopping = False
        self._consumer_task = asyncio.create_task(self._consumer_loop())
        logger.info("Redis training queue consumer started: %s", self._queue_name)

    async def stop(self) -> None:
        self._stopping = True
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
        self._consumer_task = None

    def enqueue(self, job_id: str) -> None:
        try:
            self._redis.rpush(self._queue_name, job_id)
            logger.info("Enqueued training job %s to Redis queue %s", job_id, self._queue_name)
        except Exception as exc:
            logger.warning(
                "Failed to enqueue training job %s to Redis queue, fallback local dispatch: %s",
                job_id,
                exc,
            )
            self._on_job(job_id)

    async def _consumer_loop(self) -> None:
        while not self._stopping:
            try:
                job_id = await asyncio.to_thread(self._dequeue_one)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Redis training queue consumer error: %s", exc)
                await asyncio.sleep(1)
                continue

            if job_id:
                self._on_job(job_id)

    def _dequeue_one(self) -> str | None:
        result: Any = self._redis.blpop(self._queue_name, timeout=1)
        if result is None:
            return None
        _queue_name, payload = result
        return str(payload)
