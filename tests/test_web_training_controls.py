"""Tests for web training control signal routing."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from datetime import datetime

import pytest
from fastapi import HTTPException

from med_core.web.api import training as training_api


def test_ws_control_ignores_cross_job_payload(monkeypatch) -> None:
    calls: list[tuple[str, signal.Signals]] = []

    def fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", fake_signal)

    dispatched = training_api._dispatch_ws_control_signal(
        route_job_id="job-selected",
        action="pause",
        payload_job_id="job-other",
    )

    assert dispatched is False
    assert calls == []


def test_ws_control_dispatches_matching_payload(monkeypatch) -> None:
    calls: list[tuple[str, signal.Signals]] = []

    def fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", fake_signal)

    dispatched = training_api._dispatch_ws_control_signal(
        route_job_id="job-selected",
        action="resume",
        payload_job_id="job-selected",
    )

    assert dispatched is True
    assert calls == [("job-selected", signal.SIGCONT)]


def test_ws_stop_dispatches_continue_then_terminate(monkeypatch) -> None:
    calls: list[tuple[str, signal.Signals]] = []

    def fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", fake_signal)

    dispatched = training_api._dispatch_ws_control_signal(
        route_job_id="job-selected",
        action="stop",
        payload_job_id=None,
    )

    assert dispatched is True
    assert calls == [
        ("job-selected", signal.SIGCONT),
        ("job-selected", signal.SIGTERM),
    ]


@dataclass
class _DummyJob:
    status: str
    completed_at: datetime | None = None


class _DummyDb:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_pause_training_updates_status_and_commits(monkeypatch) -> None:
    job = _DummyJob(status="running")
    db = _DummyDb()
    signal_calls: list[tuple[str, signal.Signals]] = []

    monkeypatch.setattr(training_api, "_get_job_or_404", lambda *_args, **_kwargs: job)

    def _fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        signal_calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", _fake_signal)

    payload = await training_api.pause_training(job_id="job-1", db=db)

    assert payload["message"] == "训练任务已暂停"
    assert job.status == "paused"
    assert signal_calls == [("job-1", signal.SIGSTOP)]
    assert db.commits == 1


@pytest.mark.asyncio
async def test_pause_training_rejects_non_running_status(monkeypatch) -> None:
    job = _DummyJob(status="completed")
    db = _DummyDb()
    monkeypatch.setattr(training_api, "_get_job_or_404", lambda *_args, **_kwargs: job)

    with pytest.raises(HTTPException) as exc_info:
        await training_api.pause_training(job_id="job-1", db=db)

    assert exc_info.value.status_code == 400
    assert "只能暂停正在运行的任务" in str(exc_info.value.detail)
    assert db.commits == 0


@pytest.mark.asyncio
async def test_resume_training_updates_status_and_commits(monkeypatch) -> None:
    job = _DummyJob(status="paused")
    db = _DummyDb()
    signal_calls: list[tuple[str, signal.Signals]] = []
    monkeypatch.setattr(training_api, "_get_job_or_404", lambda *_args, **_kwargs: job)

    def _fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        signal_calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", _fake_signal)

    payload = await training_api.resume_training(job_id="job-1", db=db)

    assert payload["message"] == "训练任务已恢复"
    assert job.status == "running"
    assert signal_calls == [("job-1", signal.SIGCONT)]
    assert db.commits == 1


@pytest.mark.asyncio
async def test_stop_training_updates_status_and_records_completion(monkeypatch) -> None:
    job = _DummyJob(status="paused")
    db = _DummyDb()
    signal_calls: list[tuple[str, signal.Signals]] = []
    monkeypatch.setattr(training_api, "_get_job_or_404", lambda *_args, **_kwargs: job)

    def _fake_signal(job_id: str, process_signal: signal.Signals) -> bool:
        signal_calls.append((job_id, process_signal))
        return True

    monkeypatch.setattr(training_api, "_signal_process", _fake_signal)

    payload = await training_api.stop_training(job_id="job-1", db=db)

    assert payload["message"] == "训练任务已停止"
    assert job.status == "stopped"
    assert job.completed_at is not None
    assert signal_calls == [
        ("job-1", signal.SIGCONT),
        ("job-1", signal.SIGTERM),
    ]
    assert db.commits == 1


@pytest.mark.asyncio
async def test_stop_training_rejects_completed_job(monkeypatch) -> None:
    job = _DummyJob(status="completed")
    db = _DummyDb()
    monkeypatch.setattr(training_api, "_get_job_or_404", lambda *_args, **_kwargs: job)

    with pytest.raises(HTTPException) as exc_info:
        await training_api.stop_training(job_id="job-1", db=db)

    assert exc_info.value.status_code == 400
    assert "无法停止该任务" in str(exc_info.value.detail)
    assert db.commits == 0
