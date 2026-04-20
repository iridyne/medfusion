"""Experiments API tests with real DB-backed records."""

from __future__ import annotations

import os
import tempfile
import uuid

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-experiments-test-")
)

from med_core.web.app import app
from med_core.web.config import settings
from med_core.web.database import SessionLocal, init_db
from med_core.web.models import ModelInfo


@pytest.fixture(scope="module", autouse=True)
def _prepare_web_storage() -> None:
    settings.initialize_directories()
    init_db()


@pytest.fixture
async def api_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client


def _create_model(name: str, tags: list[str] | None = None) -> int:
    db = SessionLocal()
    try:
        row = ModelInfo(
            name=name,
            description="experiments api test model",
            model_type="classification",
            architecture="resnet18",
            checkpoint_path=f"/tmp/{name}.pth",
            metrics={
                "accuracy": 0.88,
                "precision": 0.84,
                "recall": 0.82,
                "f1_score": 0.83,
                "auc_roc": 0.91,
                "loss": 0.2,
            },
            accuracy=0.88,
            loss=0.2,
            training_time=180.0,
            trained_epochs=5,
            tags=tags,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


def _fetch_model(model_id: int) -> ModelInfo | None:
    db = SessionLocal()
    try:
        return db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
    finally:
        db.close()


@pytest.mark.asyncio
async def test_experiments_list_uses_real_model_records(api_client) -> None:
    marker = f"exp-api-{uuid.uuid4().hex[:8]}"
    model_id = _create_model(marker)

    response = await api_client.get("/api/experiments/")
    assert response.status_code == 200
    payload = response.json()

    assert payload["total"] >= 1
    matched = next((item for item in payload["experiments"] if item["name"] == marker), None)
    assert matched is not None
    assert matched["id"] == f"model-{model_id}"
    assert matched["status"] == "completed"
    assert matched["metrics"]["accuracy"] == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_experiments_favorite_toggle_and_delete_for_real_model(api_client) -> None:
    marker = f"exp-api-toggle-{uuid.uuid4().hex[:8]}"
    model_id = _create_model(marker)
    experiment_id = f"model-{model_id}"

    favorite_response = await api_client.patch(f"/api/experiments/{experiment_id}/favorite")
    assert favorite_response.status_code == 200
    assert favorite_response.json()["success"] is True
    assert favorite_response.json()["is_favorite"] is True

    refreshed = _fetch_model(model_id)
    assert refreshed is not None
    tags = refreshed.tags if isinstance(refreshed.tags, list) else []
    assert "favorite" in tags

    delete_response = await api_client.delete(f"/api/experiments/{experiment_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True
    assert _fetch_model(model_id) is None
