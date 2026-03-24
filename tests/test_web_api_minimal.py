"""Minimal Web API contract tests for dataset/model/training routes."""

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-web-test-"))

from med_core.web.app import app
from med_core.web.config import settings
from med_core.web.database import init_db
from test_build_results import _create_checkpoint_and_logs


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


async def test_web_basic_routes(api_client) -> None:
    mock_dataset_path = Path(__file__).resolve().parents[1] / "data" / "mock"

    # Health
    health = await api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    # Dataset CRUD + analyze
    created_dataset = await api_client.post(
        "/api/datasets/",
        json={
            "name": "ci-dataset",
            "data_path": str(mock_dataset_path),
            "dataset_type": "multimodal",
            "num_samples": 30,
            "num_classes": 2,
        },
    )
    assert created_dataset.status_code == 200
    dataset_id = created_dataset.json()["id"]

    dataset_list = await api_client.get("/api/datasets/")
    assert dataset_list.status_code == 200
    assert isinstance(dataset_list.json(), list)

    dataset_stats = await api_client.get("/api/datasets/statistics")
    assert dataset_stats.status_code == 200
    assert "total_datasets" in dataset_stats.json()

    dataset_analysis = await api_client.post(f"/api/datasets/{dataset_id}/analyze")
    assert dataset_analysis.status_code == 200

    # Model CRUD
    created_model = await api_client.post(
        "/api/models/",
        json={
            "name": "ci-model",
            "backbone": "resnet18",
            "num_classes": 2,
            "accuracy": 0.88,
        },
    )
    assert created_model.status_code == 200
    model_id = created_model.json()["id"]

    model_list = await api_client.get("/api/models/")
    assert model_list.status_code == 200
    assert isinstance(model_list.json(), list)

    model_stats = await api_client.get("/api/models/statistics")
    assert model_stats.status_code == 200
    assert "total_models" in model_stats.json()

    # Training lifecycle (start + status)
    started_job = await api_client.post(
        "/api/training/start",
        json={
            "experiment_name": "ci-exp",
            "training_model_config": {
                "backbone": "mobilenetv2",
                "num_classes": 2,
                "pretrained": False,
            },
            "dataset_config": {
                "dataset": "ci-dataset",
                "data_path": str(mock_dataset_path),
                "num_classes": 2,
            },
            "training_config": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "image_size": 64,
                "num_workers": 0,
                "mixed_precision": False,
            },
        },
    )
    assert started_job.status_code == 200
    job_id = started_job.json()["job_id"]

    job_status = await api_client.get(f"/api/training/{job_id}/status")
    assert job_status.status_code == 200
    assert job_status.json()["job_id"] == job_id

    for _ in range(45):
        current_status = await api_client.get(f"/api/training/{job_id}/status")
        assert current_status.status_code == 200
        if current_status.json()["status"] == "completed":
            break
        if current_status.json()["status"] == "failed":
            raise AssertionError(current_status.json().get("error_message"))
        await asyncio.sleep(1.0)
    else:
        raise AssertionError("training job did not complete in expected time")

    history_payload = await api_client.get(f"/api/training/{job_id}/history")
    assert history_payload.status_code == 200
    assert len(history_payload.json()["entries"]) >= 1

    refreshed_models = await api_client.get("/api/models/")
    assert refreshed_models.status_code == 200
    generated_model = next(
        item for item in refreshed_models.json() if item["name"] == "ci-exp-model"
    )
    assert generated_model["validation"]["overview"]["sample_count"] > 0
    assert len(generated_model["validation"]["per_class"]) == 2
    assert generated_model["metrics"]["balanced_accuracy"] >= 0
    assert any(
        artifact["key"] == "validation" for artifact in generated_model["result_files"]
    )

    generated_model_detail = await api_client.get(f"/api/models/{generated_model['id']}")
    assert generated_model_detail.status_code == 200
    detail_payload = generated_model_detail.json()
    assert detail_payload["validation"]["prediction_summary"]["error_count"] >= 0
    assert detail_payload["validation"]["dataset"]["num_classes"] == 2

    # Cleanup
    deleted_model = await api_client.delete(f"/api/models/{model_id}")
    assert deleted_model.status_code == 200

    deleted_generated_model = await api_client.delete(f"/api/models/{generated_model['id']}")
    assert deleted_generated_model.status_code == 200

    deleted_dataset = await api_client.delete(f"/api/datasets/{dataset_id}")
    assert deleted_dataset.status_code == 200


async def test_web_can_import_real_cli_run(api_client, tmp_path) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    imported = await api_client.post(
        "/api/models/import-run",
        json={
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "split": "train",
            "attention_samples": 2,
            "survival_time_column": "survival_time",
            "survival_event_column": "event",
            "importance_sample_limit": 8,
            "name": "ci-imported-real-run",
            "tags": ["ci-import"],
        },
    )
    assert imported.status_code == 200
    payload = imported.json()
    assert payload["name"] == "ci-imported-real-run"
    assert payload["validation"]["overview"]["split"] == "train"
    assert payload["visualizations"]["confusion_matrix"]["plot_url"]
    assert payload["visualizations"]["attention_maps"]
    assert payload["validation"]["survival"]["c_index"] is not None
    assert payload["visualizations"]["survival_curve"]["image_url"]
    assert payload["visualizations"]["risk_score_distribution"]["image_url"]
    assert payload["validation"]["global_feature_importance"]["top_features"]
    assert payload["visualizations"]["feature_importance_bar"]["image_url"]
    assert payload["visualizations"]["feature_importance_beeswarm"]["image_url"]
    assert any(artifact["key"] == "survival" for artifact in payload["result_files"])
    assert any(artifact["key"] == "feature_importance" for artifact in payload["result_files"])
    assert any(artifact["key"] == "report" for artifact in payload["result_files"])

    listed = await api_client.get("/api/models/")
    assert listed.status_code == 200
    assert any(item["id"] == payload["id"] for item in listed.json())

    deleted = await api_client.delete(f"/api/models/{payload['id']}")
    assert deleted.status_code == 200
