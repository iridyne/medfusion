"""Minimal contract tests for project workspace APIs."""

import os
import tempfile
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-projects-test-"))

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


async def test_projects_templates_and_crud(api_client, tmp_path: Path) -> None:
    templates_response = await api_client.get("/api/projects/templates")
    assert templates_response.status_code == 200
    templates = templates_response.json()["templates"]
    assert any(item["id"] == "binary_basic" for item in templates)

    dataset_response = await api_client.post(
        "/api/datasets/",
        json={
            "name": "project-dataset",
            "data_path": str(tmp_path),
            "dataset_type": "multimodal",
            "num_samples": 12,
            "num_classes": 2,
        },
    )
    assert dataset_response.status_code == 200
    dataset_id = dataset_response.json()["id"]

    created_project = await api_client.post(
        "/api/projects/",
        json={
            "name": "binary-project",
            "description": "project workspace smoke test",
            "task_type": "binary_classification",
            "template_id": "binary_basic",
            "dataset_id": dataset_id,
            "tags": ["ci", "local-pro"],
        },
    )
    assert created_project.status_code == 200
    project_payload = created_project.json()
    project_id = project_payload["id"]
    assert project_payload["dataset_id"] == dataset_id
    assert project_payload["job_count"] == 0

    listed_projects = await api_client.get("/api/projects/")
    assert listed_projects.status_code == 200
    assert any(item["id"] == project_id for item in listed_projects.json())

    updated_project = await api_client.patch(
        f"/api/projects/{project_id}",
        json={
            "status": "running",
            "output_dir": str(tmp_path / "outputs"),
        },
    )
    assert updated_project.status_code == 200
    assert updated_project.json()["status"] == "running"

    project_detail = await api_client.get(f"/api/projects/{project_id}")
    assert project_detail.status_code == 200
    detail_payload = project_detail.json()
    assert detail_payload["template_id"] == "binary_basic"
    assert detail_payload["jobs"] == []
    assert detail_payload["models"] == []

    export_dir = tmp_path / "outputs"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "summary.json").write_text('{"ok": true}', encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: binary-project\n", encoding="utf-8")

    updated_exportable = await api_client.patch(
        f"/api/projects/{project_id}",
        json={
          "config_path": str(config_path),
          "output_dir": str(export_dir),
          "status": "completed",
        },
    )
    assert updated_exportable.status_code == 200

    exported = await api_client.post(f"/api/projects/{project_id}/export")
    assert exported.status_code == 200
    download_url = exported.json()["download_url"]
    assert download_url

    downloaded = await api_client.get(download_url)
    assert downloaded.status_code == 200
    assert downloaded.content

    deleted_project = await api_client.delete(f"/api/projects/{project_id}")
    assert deleted_project.status_code == 200

    deleted_dataset = await api_client.delete(f"/api/datasets/{dataset_id}")
    assert deleted_dataset.status_code == 200
