"""Experiments API tests with real DB-backed records."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-experiments-test-")
)

from med_core.web.app import app
from med_core.web.config import settings
from med_core.web.database import SessionLocal, init_db
from med_core.web.models import Experiment as ExperimentRecord
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


def _create_model(
    name: str,
    tags: list[str] | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> int:
    db = SessionLocal()
    try:
        config: dict[str, object] | None = None
        if artifact_paths:
            config = {"artifact_paths": artifact_paths}

        row = ModelInfo(
            name=name,
            description="experiments api test model",
            model_type="classification",
            architecture="resnet18",
            checkpoint_path=f"/tmp/{name}.pth",
            config=config,
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


def _delete_model(model_id: int) -> None:
    db = SessionLocal()
    try:
        row = db.query(ModelInfo).filter(ModelInfo.id == model_id).first()
        if row is not None:
            db.delete(row)
            db.commit()
    finally:
        db.close()


def _create_experiment_record(
    name: str,
    artifact_paths: dict[str, str] | None = None,
) -> int:
    db = SessionLocal()
    try:
        config: dict[str, object] = {
            "model": {"backbone": "resnet18", "fusion_type": "concatenate"},
            "training": {
                "num_epochs": 4,
                "optimizer": {"optimizer": "adam", "learning_rate": 0.001},
            },
            "data": {"batch_size": 16},
        }
        if artifact_paths:
            config["artifact_paths"] = artifact_paths

        row = ExperimentRecord(
            name=name,
            description="experiments api test run record",
            config=config,
            status="completed",
            metrics={
                "accuracy": 0.89,
                "precision": 0.87,
                "recall": 0.86,
                "f1_score": 0.865,
                "auc": 0.92,
                "loss": 0.19,
            },
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return int(row.id)
    finally:
        db.close()


def _delete_experiment_record(experiment_id: int) -> None:
    db = SessionLocal()
    try:
        row = db.query(ExperimentRecord).filter(ExperimentRecord.id == experiment_id).first()
        if row is not None:
            db.delete(row)
            db.commit()
    finally:
        db.close()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


@pytest.mark.asyncio
async def test_experiments_metrics_history_prefers_artifact_history(
    api_client, tmp_path: Path,
) -> None:
    marker = f"exp-api-history-{uuid.uuid4().hex[:8]}"
    history_path = tmp_path / "logs" / "history.json"
    _write_json(
        history_path,
        {
            "entries": [
                {
                    "epoch": 1,
                    "train_loss": 0.93,
                    "val_loss": 0.81,
                    "train_accuracy": 0.66,
                    "val_accuracy": 0.7,
                    "learning_rate": 0.001,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.74,
                    "val_loss": 0.63,
                    "train_accuracy": 0.78,
                    "val_accuracy": 0.8,
                    "learning_rate": 0.0008,
                },
            ]
        },
    )
    model_id = _create_model(
        marker,
        artifact_paths={"history_path": str(history_path)},
    )

    try:
        response = await api_client.get(f"/api/experiments/model-{model_id}/metrics")
        assert response.status_code == 200
        payload = response.json()
        assert payload["experiment_id"] == f"model-{model_id}"
        assert len(payload["history"]) == 2
        assert payload["history"][0]["epoch"] == 1
        assert payload["history"][0]["train_loss"] == pytest.approx(0.93)
        assert payload["history"][1]["val_accuracy"] == pytest.approx(0.8)
    finally:
        _delete_model(model_id)


@pytest.mark.asyncio
async def test_experiments_confusion_matrix_prefers_artifact_json(
    api_client, tmp_path: Path,
) -> None:
    marker = f"exp-api-cm-{uuid.uuid4().hex[:8]}"
    confusion_path = tmp_path / "metrics" / "confusion_matrix.json"
    _write_json(
        confusion_path,
        {
            "labels": ["negative", "positive"],
            "matrix": [[18, 2], [3, 17]],
        },
    )
    model_id = _create_model(
        marker,
        artifact_paths={"confusion_matrix_json_path": str(confusion_path)},
    )

    try:
        response = await api_client.get(
            f"/api/experiments/model-{model_id}/confusion-matrix"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["classes"] == ["negative", "positive"]
        assert payload["matrix"] == [[18, 2], [3, 17]]
        assert payload["total"] == 40
    finally:
        _delete_model(model_id)


@pytest.mark.asyncio
async def test_experiments_roc_curve_prefers_artifact_json(
    api_client, tmp_path: Path,
) -> None:
    marker = f"exp-api-roc-{uuid.uuid4().hex[:8]}"
    roc_path = tmp_path / "metrics" / "roc_curve.json"
    _write_json(
        roc_path,
        {
            "auc": 0.9731,
            "points": [
                {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
                [0.12, 0.78, 0.88],
                [1.0, 1.0],
            ],
        },
    )
    model_id = _create_model(
        marker,
        artifact_paths={"roc_curve_json_path": str(roc_path)},
    )

    try:
        response = await api_client.get(f"/api/experiments/model-{model_id}/roc-curve")
        assert response.status_code == 200
        payload = response.json()
        assert payload["auc"] == pytest.approx(0.9731)
        assert len(payload["points"]) == 3
        assert payload["points"][1]["fpr"] == pytest.approx(0.12)
        assert payload["points"][1]["tpr"] == pytest.approx(0.78)
        assert payload["points"][1]["threshold"] == pytest.approx(0.88)
        assert payload["points"][2]["threshold"] == pytest.approx(0.0)
    finally:
        _delete_model(model_id)


@pytest.mark.asyncio
async def test_run_experiment_metrics_uses_experiment_artifact_paths_when_model_missing(
    api_client, tmp_path: Path,
) -> None:
    marker = f"exp-api-run-history-{uuid.uuid4().hex[:8]}"
    history_path = tmp_path / "run-metrics" / "history.json"
    _write_json(
        history_path,
        {
            "entries": [
                {
                    "epoch": 1,
                    "train_loss": 0.51,
                    "val_loss": 0.43,
                    "train_accuracy": 0.82,
                    "val_accuracy": 0.84,
                    "learning_rate": 0.0005,
                }
            ]
        },
    )

    run_id = _create_experiment_record(
        marker,
        artifact_paths={"history_path": str(history_path)},
    )

    try:
        response = await api_client.get(f"/api/experiments/run-{run_id}/metrics")
        assert response.status_code == 200
        payload = response.json()
        assert payload["experiment_id"] == f"run-{run_id}"
        assert len(payload["history"]) == 1
        assert payload["history"][0]["train_loss"] == pytest.approx(0.51)
        assert payload["history"][0]["val_accuracy"] == pytest.approx(0.84)
    finally:
        _delete_experiment_record(run_id)


@pytest.mark.asyncio
async def test_run_experiment_charts_use_experiment_artifact_paths_when_model_missing(
    api_client, tmp_path: Path,
) -> None:
    marker = f"exp-api-run-chart-{uuid.uuid4().hex[:8]}"
    confusion_path = tmp_path / "run-metrics" / "confusion_matrix.json"
    roc_path = tmp_path / "run-metrics" / "roc_curve.json"
    _write_json(
        confusion_path,
        {
            "labels": ["benign", "malignant"],
            "matrix": [[22, 1], [2, 19]],
        },
    )
    _write_json(
        roc_path,
        {
            "auc": 0.955,
            "points": [
                {"fpr": 0.0, "tpr": 0.0, "threshold": 1.0},
                {"fpr": 0.2, "tpr": 0.9, "threshold": 0.7},
                {"fpr": 1.0, "tpr": 1.0, "threshold": 0.0},
            ],
        },
    )
    run_id = _create_experiment_record(
        marker,
        artifact_paths={
            "confusion_matrix_json_path": str(confusion_path),
            "roc_curve_json_path": str(roc_path),
        },
    )

    try:
        confusion_response = await api_client.get(
            f"/api/experiments/run-{run_id}/confusion-matrix"
        )
        assert confusion_response.status_code == 200
        confusion_payload = confusion_response.json()
        assert confusion_payload["classes"] == ["benign", "malignant"]
        assert confusion_payload["matrix"] == [[22, 1], [2, 19]]

        roc_response = await api_client.get(f"/api/experiments/run-{run_id}/roc-curve")
        assert roc_response.status_code == 200
        roc_payload = roc_response.json()
        assert roc_payload["auc"] == pytest.approx(0.955)
        assert len(roc_payload["points"]) == 3
        assert roc_payload["points"][1]["fpr"] == pytest.approx(0.2)
        assert roc_payload["points"][1]["tpr"] == pytest.approx(0.9)
    finally:
        _delete_experiment_record(run_id)
