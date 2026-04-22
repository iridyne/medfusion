"""Workflow API boundary tests."""

import os
import tempfile

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-workflow-test-"))

from med_core.web.app import app
from med_core.web.config import settings
from med_core.web.database import init_db


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


async def test_system_features_report_workflow_as_preview(api_client) -> None:
    response = await api_client.get("/api/system/features")
    assert response.status_code == 200

    payload = response.json()
    deployment_modes = payload["deployment_modes"]
    assert [item["id"] for item in deployment_modes] == [
        "local_browser",
        "private_server",
        "managed_cloud",
    ]
    assert all(item["api_bff"] == "FastAPI" for item in deployment_modes)
    assert all(item["same_capabilities_as_runtime"] is True for item in deployment_modes)
    assert payload["advanced_builder"]["status"] == "preview"
    assert payload["advanced_builder"]["route"] == "/config/advanced"
    assert payload["advanced_builder"]["canvas_route"] == "/config/advanced/canvas"
    assert payload["advanced_builder"]["default_entry"] is False
    assert "fusion" in payload["advanced_builder"]["supported_families"]
    assert payload["auth"]["enabled"] is False
    assert payload["auth"]["mode"] == "disabled"
    assert payload["auth"]["jwt_runtime_available"] in {True, False}
    assert payload["auth"]["rbac_roles"] == ["viewer", "operator", "admin"]
    assert payload["training_queue"]["backend"] == "local"
    assert payload["training_queue"]["queue_name"] == "medfusion:training:jobs"
    assert payload["training_queue"]["status"] == "local_default"
    assert payload["workflow"]["enabled"] is True
    assert payload["workflow"]["status"] == "preview"
    assert payload["workflow"]["ui_exposed"] is True
    assert payload["recommended_primary_flow"] == [
        "workbench",
        "run_wizard",
        "training_monitor",
        "model_library",
    ]


async def test_workflow_validate_reports_preview_scope_errors(api_client) -> None:
    response = await api_client.post(
        "/api/workflows/validate",
        json={
            "workflow": {
                "nodes": [],
                "edges": [],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["valid"] is False
    assert any("工作流为空" in error for error in payload["errors"])


async def test_workflow_execute_starts_real_training_job_preview(
    monkeypatch, api_client
) -> None:
    from med_core.web.api import training as training_api
    from med_core.web.routers import workflow as workflow_router

    captured_payload: dict[str, object] = {}

    async def _fake_start_training(config, _db):
        captured_payload.update(config.model_dump())
        return {
            "job_id": "job-workflow-preview",
            "status": "running",
            "message": "训练任务已启动",
        }

    monkeypatch.setattr(training_api, "start_training", _fake_start_training)

    workflow_router.active_workflows.clear()

    response = await api_client.post(
        "/api/workflows/execute",
        json={
            "name": "preview-workflow",
            "workflow": {
                "nodes": [
                    {
                        "id": "data_1",
                        "type": "dataLoader",
                        "data": {
                            "datasetName": "repo-mock",
                            "dataPath": "data/mock",
                            "csvPath": "data/mock/metadata.csv",
                            "imageDir": "data/mock",
                            "imagePathColumn": "image_path",
                            "targetColumn": "diagnosis",
                            "batchSize": 8,
                            "numWorkers": 0,
                        },
                        "position": {"x": 0, "y": 0},
                    },
                    {
                        "id": "model_1",
                        "type": "model",
                        "data": {
                            "backbone": "resnet18",
                            "numClasses": 2,
                            "pretrained": False,
                            "fusion": "concatenate",
                        },
                        "position": {"x": 200, "y": 0},
                    },
                    {
                        "id": "train_1",
                        "type": "training",
                        "data": {
                            "epochs": 3,
                            "learningRate": 0.001,
                            "optimizer": "adam",
                        },
                        "position": {"x": 400, "y": 0},
                    },
                ],
                "edges": [
                    {
                        "id": "e1",
                        "source": "model_1",
                        "target": "train_1",
                        "sourceHandle": "model",
                        "targetHandle": "model",
                    },
                    {
                        "id": "e2",
                        "source": "data_1",
                        "target": "train_1",
                        "sourceHandle": "dataset",
                        "targetHandle": "train_data",
                    },
                ],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"
    assert payload["training_job_id"] == "job-workflow-preview"
    assert captured_payload["experiment_name"] == "preview-workflow"
    assert captured_payload["source_context"]["source_type"] == "workflow"
    assert captured_payload["source_context"]["entrypoint"] == "workflow-editor"
