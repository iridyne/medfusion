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


async def test_system_features_report_workflow_as_disabled(api_client) -> None:
    response = await api_client.get("/api/system/features")
    assert response.status_code == 200

    payload = response.json()
    assert payload["workflow"]["enabled"] is False
    assert payload["workflow"]["status"] == "disabled"
    assert payload["workflow"]["ui_exposed"] is False
    assert payload["recommended_primary_flow"] == [
        "workbench",
        "run_wizard",
        "training_monitor",
        "model_library",
    ]


async def test_workflow_validate_returns_experimental_disabled(api_client) -> None:
    response = await api_client.post(
        "/api/workflows/validate",
        json={
            "workflow": {
                "nodes": [],
                "edges": [],
            },
        },
    )

    assert response.status_code == 503
    payload = response.json()["detail"]
    assert payload["code"] == "workflow_experimental_disabled"
    assert payload["enable_env"] == "MEDFUSION_ENABLE_EXPERIMENTAL_WORKFLOW=true"
