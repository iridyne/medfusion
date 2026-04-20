"""Web SPA route fallback contract tests."""

import os
import tempfile

import httpx
import pytest

pytest.importorskip("fastapi")

os.environ.setdefault("MEDFUSION_DATA_DIR", tempfile.mkdtemp(prefix="medfusion-web-test-"))

from med_core.web.app import app


@pytest.mark.asyncio
async def test_start_route_is_served_by_spa_fallback() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        health = await client.get("/health")
        assert health.status_code == 200

        start_page = await client.get("/start")
        assert start_page.status_code == 200
