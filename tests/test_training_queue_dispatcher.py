"""Training queue dispatcher selection tests."""

from __future__ import annotations

import pytest

from med_core.web.api import training as training_api
from med_core.web.queue_dispatcher import LocalTrainingQueueDispatcher


def _settings_snapshot() -> dict[str, object]:
    return {
        "training_queue_backend": training_api.settings.training_queue_backend,
        "redis_url": training_api.settings.redis_url,
        "redis_queue_name": training_api.settings.redis_queue_name,
    }


def _restore_settings(snapshot: dict[str, object]) -> None:
    training_api.settings.training_queue_backend = str(snapshot["training_queue_backend"])
    training_api.settings.redis_url = (
        str(snapshot["redis_url"]) if snapshot["redis_url"] is not None else None
    )
    training_api.settings.redis_queue_name = str(snapshot["redis_queue_name"])


def test_training_queue_dispatcher_defaults_to_local() -> None:
    snapshot = _settings_snapshot()
    previous_dispatcher = training_api._queue_dispatcher
    try:
        training_api.settings.training_queue_backend = "local"
        training_api._queue_dispatcher = None
        dispatcher = training_api._training_queue_dispatcher()
        assert isinstance(dispatcher, LocalTrainingQueueDispatcher)
    finally:
        _restore_settings(snapshot)
        training_api._queue_dispatcher = previous_dispatcher


def test_training_queue_dispatcher_fallbacks_to_local_when_redis_unavailable(
    monkeypatch,
) -> None:
    snapshot = _settings_snapshot()
    previous_dispatcher = training_api._queue_dispatcher
    try:
        training_api.settings.training_queue_backend = "redis"
        training_api.settings.redis_url = "redis://127.0.0.1:6379/0"
        training_api._queue_dispatcher = None

        class _BrokenRedisDispatcher:
            def __init__(self, **kwargs):
                raise RuntimeError("redis unavailable in test")

        monkeypatch.setattr(
            training_api,
            "RedisTrainingQueueDispatcher",
            _BrokenRedisDispatcher,
        )

        dispatcher = training_api._training_queue_dispatcher()
        assert isinstance(dispatcher, LocalTrainingQueueDispatcher)
    finally:
        _restore_settings(snapshot)
        training_api._queue_dispatcher = previous_dispatcher


@pytest.mark.asyncio
async def test_startup_and_shutdown_queue_dispatcher_lifecycle() -> None:
    snapshot = _settings_snapshot()
    previous_dispatcher = training_api._queue_dispatcher
    try:
        training_api.settings.training_queue_backend = "local"
        training_api._queue_dispatcher = None
        await training_api.startup_training_queue_dispatcher()
        assert isinstance(training_api._queue_dispatcher, LocalTrainingQueueDispatcher)
        await training_api.shutdown_training_queue_dispatcher()
        assert training_api._queue_dispatcher is None
    finally:
        _restore_settings(snapshot)
        training_api._queue_dispatcher = previous_dispatcher

