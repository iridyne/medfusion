"""Auth + RBAC API behavior tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-auth-test-"))

from med_core.web.app import app
from med_core.web.auth import create_access_token, is_jwt_runtime_available
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


@pytest.fixture
def _restore_auth_settings():
    snapshot = {
        "auth_enabled": settings.auth_enabled,
        "auth_token": settings.auth_token,
        "auth_username": settings.auth_username,
        "auth_password": settings.auth_password,
        "auth_default_role": settings.auth_default_role,
    }
    try:
        yield
    finally:
        settings.auth_enabled = snapshot["auth_enabled"]
        settings.auth_token = snapshot["auth_token"]
        settings.auth_username = snapshot["auth_username"]
        settings.auth_password = snapshot["auth_password"]
        settings.auth_default_role = snapshot["auth_default_role"]


async def test_static_token_mode_protects_api_routes(
    api_client,
    _restore_auth_settings,
) -> None:
    settings.auth_enabled = True
    settings.auth_token = "static-auth-token"
    settings.auth_password = None

    unauth = await api_client.get("/api/datasets/")
    assert unauth.status_code == 401

    public = await api_client.get("/api/system/features")
    assert public.status_code == 200

    authed = await api_client.get(
        "/api/datasets/",
        headers={"Authorization": "Bearer static-auth-token"},
    )
    assert authed.status_code == 200


async def test_jwt_token_endpoint_requires_bootstrap_password(
    api_client,
    _restore_auth_settings,
) -> None:
    if not is_jwt_runtime_available():
        pytest.skip("PyJWT runtime unavailable; skip JWT token endpoint test.")

    settings.auth_enabled = True
    settings.auth_token = None
    settings.auth_username = "admin"
    settings.auth_password = "change-me"
    settings.auth_default_role = "viewer"

    response = await api_client.post(
        "/api/auth/token",
        json={"username": "admin", "password": "change-me"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["access_token"], str)
    assert payload["role"] == "viewer"


async def test_rbac_viewer_denies_write_and_operator_allows_write(
    api_client,
    _restore_auth_settings,
) -> None:
    if not is_jwt_runtime_available():
        pytest.skip("PyJWT runtime unavailable; skip JWT RBAC role test.")

    settings.auth_enabled = True
    settings.auth_token = None
    settings.auth_default_role = "admin"

    mock_dataset_path = Path(__file__).resolve().parents[1] / "data" / "mock"
    payload = {
        "name": "rbac-dataset",
        "data_path": str(mock_dataset_path),
        "dataset_type": "multimodal",
        "num_samples": 10,
        "num_classes": 2,
    }

    viewer_token = create_access_token(subject="viewer-user", role="viewer")
    viewer_resp = await api_client.post(
        "/api/datasets/",
        json=payload,
        headers={"Authorization": f"Bearer {viewer_token}"},
    )
    assert viewer_resp.status_code == 403

    operator_token = create_access_token(subject="operator-user", role="operator")
    operator_resp = await api_client.post(
        "/api/datasets/",
        json={**payload, "name": "rbac-dataset-operator"},
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert operator_resp.status_code == 200
