"""API-level tests for training control endpoints."""

from __future__ import annotations

import os
import signal
import tempfile
import uuid
from datetime import datetime

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-training-control-test-")
)

from med_core.web.app import app
from med_core.web.database import SessionLocal, init_db
from med_core.web.models import TrainingJob
from med_core.web.time_utils import utcnow


@pytest.fixture(scope="module", autouse=True)
def _prepare_web_storage() -> None:
    init_db()


@pytest.fixture
async def api_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


def _create_job(status: str) -> str:
    db = SessionLocal()
    try:
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id=job_id,
            config={"experiment_name": "control-test"},
            total_epochs=4,
            status=status,
            progress=0.0,
            current_epoch=0,
            created_at=utcnow(),
            started_at=utcnow(),
        )
        db.add(job)
        db.commit()
        return job_id
    finally:
        db.close()


def _fetch_job(job_id: str) -> TrainingJob:
    db = SessionLocal()
    try:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        assert job is not None
        db.expunge(job)
        return job
    finally:
        db.close()


@pytest.mark.asyncio
async def test_pause_and_resume_training_api(monkeypatch, api_client) -> None:
    signals: list[tuple[str, int]] = []

    def _fake_signal(job_id: str, process_signal) -> bool:
        signals.append((job_id, int(process_signal)))
        return True

    from med_core.web.api import training as training_api

    monkeypatch.setattr(training_api, "_signal_process", _fake_signal)
    job_id = _create_job("running")

    pause_response = await api_client.post(f"/api/training/{job_id}/pause")
    assert pause_response.status_code == 200
    assert pause_response.json()["message"] == "训练任务已暂停"
    paused_job = _fetch_job(job_id)
    assert paused_job.status == "paused"

    resume_response = await api_client.post(f"/api/training/{job_id}/resume")
    assert resume_response.status_code == 200
    assert resume_response.json()["message"] == "训练任务已恢复"
    resumed_job = _fetch_job(job_id)
    assert resumed_job.status == "running"

    assert signals == [
        (job_id, int(signal.SIGSTOP)),
        (job_id, int(signal.SIGCONT)),
    ]


@pytest.mark.asyncio
async def test_stop_training_api_sets_completion_time(monkeypatch, api_client) -> None:
    signals: list[tuple[str, int]] = []

    def _fake_signal(job_id: str, process_signal) -> bool:
        signals.append((job_id, int(process_signal)))
        return True

    from med_core.web.api import training as training_api

    monkeypatch.setattr(training_api, "_signal_process", _fake_signal)
    job_id = _create_job("paused")

    stop_response = await api_client.post(f"/api/training/{job_id}/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["message"] == "训练任务已停止"

    stopped_job = _fetch_job(job_id)
    assert stopped_job.status == "stopped"
    assert isinstance(stopped_job.completed_at, datetime)
    assert signals == [
        (job_id, int(signal.SIGCONT)),
        (job_id, int(signal.SIGTERM)),
    ]


@pytest.mark.asyncio
async def test_pause_rejects_completed_job(api_client) -> None:
    job_id = _create_job("completed")

    response = await api_client.post(f"/api/training/{job_id}/pause")
    assert response.status_code == 400
    assert response.json()["detail"] == "只能暂停正在运行的任务"
