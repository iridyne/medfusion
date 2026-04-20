"""ComfyUI API tests."""

from __future__ import annotations

import os
import tempfile

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault(
    "MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-comfyui-test-")
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


async def test_comfyui_health_returns_probe_result(monkeypatch, api_client) -> None:
    from med_core.web.api import comfyui as comfyui_api

    async def _fake_probe_comfyui(*, base_url: str, timeout_sec: float):
        assert base_url == "http://127.0.0.1:8188"
        assert timeout_sec == settings.comfyui_probe_timeout_sec
        return {
            "reachable": True,
            "status_code": 200,
            "latency_ms": 35.2,
            "probe_url": "http://127.0.0.1:8188/system_stats",
            "message": "ok",
            "payload_preview": {"system": {"os": "windows"}},
        }

    monkeypatch.setattr(comfyui_api, "probe_comfyui", _fake_probe_comfyui)

    response = await api_client.get("/api/comfyui/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["base_url"] == "http://127.0.0.1:8188"
    assert payload["open_url"] == "http://127.0.0.1:8188"
    assert "python main.py" in payload["recommended_start_command"]
    assert payload["probe"]["reachable"] is True
    assert payload["probe"]["status_code"] == 200


async def test_comfyui_health_rejects_invalid_base_url(api_client) -> None:
    response = await api_client.get(
        "/api/comfyui/health",
        params={"base_url": "ftp://127.0.0.1:8188"},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "invalid_comfyui_base_url"


async def test_comfyui_adapter_profiles_expose_compile_ready_blueprints(
    api_client,
) -> None:
    response = await api_client.get("/api/comfyui/adapter-profiles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "adapter_preview"
    assert "source_boundary" in payload
    assert len(payload["profiles"]) >= 1

    quickstart_profile = next(
        item for item in payload["profiles"] if item["id"] == "quickstart_multimodal"
    )
    assert quickstart_profile["target_canvas_route"].startswith(
        "/config/advanced/canvas?blueprint="
    )
    assert any(
        component["family"] == "vision_backbone"
        for component in quickstart_profile["components"]
    )
    assert quickstart_profile["default_import_prefill"]["config_path"] == (
        "configs/starter/quickstart.yaml"
    )
