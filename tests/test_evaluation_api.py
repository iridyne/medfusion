"""Independent evaluation API tests."""

import os
import tempfile

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR",
    tempfile.mkdtemp(prefix="medfusion-evaluation-web-test-"),
)

from test_build_results import _create_checkpoint_and_logs

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


async def test_independent_evaluation_can_run_without_import(api_client, tmp_path) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    response = await api_client.post(
        "/api/evaluation/run",
        json={
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "split": "test",
            "attention_samples": 2,
            "importance_sample_limit": 8,
            "import_to_model_library": False,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["mode"] == "evaluate_only"
    assert payload["validation"]["overview"]["split"] == "test"
    assert payload["summary"]["sample_count"] > 0
    assert payload["artifact_paths"]["summary_path"].endswith("summary.json")
    assert payload["model_library_import"]["imported"] is False
    assert payload["model_library_import"]["model_id"] is None


async def test_independent_evaluation_can_import_into_model_library(
    api_client,
    tmp_path,
) -> None:
    config_path, checkpoint_path = _create_checkpoint_and_logs(
        tmp_path,
        include_survival=True,
    )

    response = await api_client.post(
        "/api/evaluation/run",
        json={
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "split": "val",
            "attention_samples": 1,
            "importance_sample_limit": 4,
            "import_to_model_library": True,
            "name": "ci-evaluation-imported-model",
            "tags": ["ci-evaluation"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "evaluate_and_import"
    assert payload["model_library_import"]["imported"] is True
    assert payload["model_library_import"]["model_id"] is not None
    assert payload["model_library_import"]["model_name"] == "ci-evaluation-imported-model"

    model_id = payload["model_library_import"]["model_id"]
    imported_detail = await api_client.get(f"/api/models/{model_id}")
    assert imported_detail.status_code == 200
    detail_payload = imported_detail.json()
    assert detail_payload["config"]["import_source"] == "evaluation_api"
    assert detail_payload["config"]["source_context"]["source_type"] == "evaluation"
    assert detail_payload["config"]["source_context"]["entrypoint"] == "evaluation-center"
    assert detail_payload["config"]["source_context"]["split"] == "val"

    deleted = await api_client.delete(f"/api/models/{model_id}")
    assert deleted.status_code == 200
