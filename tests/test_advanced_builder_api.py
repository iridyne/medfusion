"""Advanced builder API tests."""

from __future__ import annotations

import os
import tempfile

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-advanced-builder-test-")
)

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


def _quickstart_graph() -> dict[str, object]:
    nodes = [
        {
            "id": "n1",
            "type": "advancedBuilderComponent",
            "position": {"x": 0, "y": 0},
            "data": {"componentId": "image_tabular_dataset"},
        },
        {
            "id": "n2",
            "type": "advancedBuilderComponent",
            "position": {"x": 1, "y": 0},
            "data": {"componentId": "resnet18_backbone"},
        },
        {
            "id": "n3",
            "type": "advancedBuilderComponent",
            "position": {"x": 1, "y": 1},
            "data": {"componentId": "mlp_tabular_encoder"},
        },
        {
            "id": "n4",
            "type": "advancedBuilderComponent",
            "position": {"x": 2, "y": 0},
            "data": {"componentId": "concatenate_fusion"},
        },
        {
            "id": "n5",
            "type": "advancedBuilderComponent",
            "position": {"x": 3, "y": 0},
            "data": {"componentId": "classification_head"},
        },
        {
            "id": "n6",
            "type": "advancedBuilderComponent",
            "position": {"x": 4, "y": 0},
            "data": {"componentId": "standard_training"},
        },
    ]
    edges = [
        {"source": "n1", "target": "n2"},
        {"source": "n1", "target": "n3"},
        {"source": "n2", "target": "n4"},
        {"source": "n3", "target": "n4"},
        {"source": "n4", "target": "n5"},
        {"source": "n5", "target": "n6"},
    ]
    return {"nodes": nodes, "edges": edges}


async def test_advanced_builder_catalog_exposes_registry(api_client) -> None:
    response = await api_client.get("/api/advanced-builder/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert "families" in payload
    assert "components" in payload
    assert "connection_rules" in payload
    assert any(component["id"] == "resnet18_backbone" for component in payload["components"])


async def test_advanced_builder_compile_returns_runspec_draft(api_client) -> None:
    response = await api_client.post(
        "/api/advanced-builder/compile",
        json=_quickstart_graph(),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["preset"] == "quickstart"
    assert payload["run_spec"] is not None
    assert payload["experiment_config"] is not None
    assert payload["contract_validation"]["ok"] is True
    assert payload["mainline_contract"]["model"]["model_type"] == "multimodal_fusion"
    assert payload["run_spec"]["model"]["vision"]["backbone"] == "resnet18"
    assert payload["run_spec"]["model"]["fusion"]["fusionType"] == "concatenate"
    assert payload["issues"] == []


async def test_advanced_builder_compile_rejects_missing_required_links(api_client) -> None:
    graph = _quickstart_graph()
    graph["edges"] = graph["edges"][:-1]
    response = await api_client.post(
        "/api/advanced-builder/compile",
        json=graph,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_spec"] is None
    assert payload["contract_validation"] is None
    assert any(issue["level"] == "error" for issue in payload["issues"])


async def test_advanced_builder_can_start_training_job(monkeypatch, api_client) -> None:
    from med_core.web.api import training as training_api
    captured_source_context: dict[str, object] = {}

    async def _fake_start_training(_config, _db):
        captured_source_context.update(_config.source_context or {})
        return {
            "job_id": "advanced-job-123",
            "status": "running",
            "message": "训练任务已启动",
        }

    monkeypatch.setattr(training_api, "start_training", _fake_start_training)

    response = await api_client.post(
        "/api/advanced-builder/start-training",
        json={**_quickstart_graph(), "blueprint_id": "quickstart_multimodal"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "advanced-job-123"
    assert payload["preset"] == "quickstart"
    assert payload["experiment_name"] == "advanced-quickstart-graph"
    assert payload["compile_result"]["contract_validation"]["ok"] is True
    assert captured_source_context["source_type"] == "advanced_builder"
    assert captured_source_context["entrypoint"] == "advanced-builder-canvas"
    assert captured_source_context["blueprint_id"] == "quickstart_multimodal"
