"""API contract tests for model artifact downloads."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-model-artifact-test-")
)

from med_core.web.app import app
from med_core.web.database import SessionLocal, init_db
from med_core.web.models import ModelInfo


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


def _create_model_with_missing_artifact(tmp_path: Path) -> int:
    checkpoint_path = tmp_path / f"checkpoint-{uuid.uuid4().hex}.pth"
    checkpoint_path.write_bytes(b"checkpoint")
    missing_report_path = tmp_path / "reports" / "report.md"

    db = SessionLocal()
    try:
        model = ModelInfo(
            name="artifact-contract-model",
            description="",
            model_type="classification",
            architecture="resnet18",
            config={
                "artifact_paths": {
                    "report_path": str(missing_report_path),
                }
            },
            metrics={"accuracy": 0.5},
            accuracy=0.5,
            loss=0.7,
            checkpoint_path=str(checkpoint_path),
        )
        db.add(model)
        db.commit()
        db.refresh(model)
        return int(model.id)
    finally:
        db.close()


def _delete_model(model_id: int) -> None:
    db = SessionLocal()
    try:
        model = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
        if model is not None:
            db.delete(model)
            db.commit()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_download_artifact_returns_404_for_missing_model(api_client) -> None:
    response = await api_client.get("/api/models/999999/artifacts/report")
    assert response.status_code == 404
    assert response.json()["detail"] == "模型不存在"


@pytest.mark.asyncio
async def test_download_artifact_returns_404_for_unknown_key(
    api_client, tmp_path
) -> None:
    model_id = _create_model_with_missing_artifact(tmp_path)
    try:
        response = await api_client.get(f"/api/models/{model_id}/artifacts/unknown-key")
        assert response.status_code == 404
        assert response.json()["detail"] == "结果文件不存在"
    finally:
        _delete_model(model_id)


@pytest.mark.asyncio
async def test_download_artifact_returns_404_for_missing_file(
    api_client, tmp_path
) -> None:
    model_id = _create_model_with_missing_artifact(tmp_path)
    try:
        response = await api_client.get(f"/api/models/{model_id}/artifacts/report")
        assert response.status_code == 404
        assert response.json()["detail"] == "结果文件不存在"
    finally:
        _delete_model(model_id)
