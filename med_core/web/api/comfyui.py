"""ComfyUI integration APIs."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Query

from ..application.advanced_builder import (
    ADVANCED_BUILDER_BLUEPRINTS,
    ADVANCED_BUILDER_COMPONENTS,
    ADVANCED_BUILDER_FAMILY_LABELS,
)
from ..config import settings

router = APIRouter()

_COMFYUI_HEALTH_PATH = "/system_stats"
_COMFYUI_START_COMMAND = "python main.py --listen 127.0.0.1 --port 8188"
_COMPONENTS_BY_ID = {
    component.id: component
    for component in ADVANCED_BUILDER_COMPONENTS
}
_DEFAULT_PREFILL_BY_BLUEPRINT: dict[str, dict[str, Any]] = {
    "quickstart_multimodal": {
        "config_path": "configs/starter/quickstart.yaml",
        "checkpoint_path": "outputs/quickstart/checkpoints/best.pth",
        "output_dir": "outputs/quickstart",
        "split": "test",
        "attention_samples": 4,
        "importance_sample_limit": 128,
    },
    "clinical_gated_baseline": {
        "config_path": "configs/starter/quickstart.yaml",
        "checkpoint_path": "outputs/medfusion-formal/advanced-clinical-graph/checkpoints/best.pth",
        "output_dir": "outputs/medfusion-formal/advanced-clinical-graph",
        "split": "test",
        "attention_samples": 4,
        "importance_sample_limit": 128,
    },
    "attention_audit_path": {
        "config_path": "configs/starter/quickstart.yaml",
        "checkpoint_path": "outputs/medfusion-formal/advanced-showcase-graph/checkpoints/best.pth",
        "output_dir": "outputs/medfusion-formal/advanced-showcase-graph",
        "split": "test",
        "attention_samples": 4,
        "importance_sample_limit": 128,
    },
}


def _normalize_base_url(raw: str) -> str:
    candidate = raw.strip().rstrip("/")
    parsed = urlparse(candidate)
    if not candidate or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_comfyui_base_url",
                "message": "ComfyUI 地址无效，请使用 http(s)://host:port 形式。",
            },
        )
    return candidate


async def probe_comfyui(
    *,
    base_url: str,
    timeout_sec: float,
) -> dict[str, Any]:
    probe_url = f"{base_url}{_COMFYUI_HEALTH_PATH}"
    started_at = perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(probe_url)
        latency_ms = round((perf_counter() - started_at) * 1000, 1)
    except httpx.RequestError as exc:
        return {
            "reachable": False,
            "status_code": None,
            "latency_ms": None,
            "probe_url": probe_url,
            "message": f"未连通 ComfyUI：{exc.__class__.__name__}",
        }

    payload_preview: dict[str, Any] | None = None
    try:
        body = response.json()
        if isinstance(body, dict):
            payload_preview = body
    except ValueError:
        payload_preview = None

    reachable = response.status_code == 200
    return {
        "reachable": reachable,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "probe_url": probe_url,
        "message": "ComfyUI 已连通，可直接进入工作流画布。"
        if reachable
        else "ComfyUI 返回异常状态，请检查进程和端口。",
        "payload_preview": payload_preview,
    }


@router.get("/health")
async def get_comfyui_health(
    base_url: str | None = Query(
        default=None,
        description="Optional ComfyUI base URL. Example: http://127.0.0.1:8188",
    ),
) -> dict[str, Any]:
    """Probe ComfyUI service and return a user-facing readiness payload."""
    resolved_base_url = _normalize_base_url(base_url or settings.comfyui_base_url)
    probe = await probe_comfyui(
        base_url=resolved_base_url,
        timeout_sec=settings.comfyui_probe_timeout_sec,
    )
    return {
        "base_url": resolved_base_url,
        "open_url": resolved_base_url,
        "recommended_start_command": _COMFYUI_START_COMMAND,
        "probe": probe,
        "handoff_hint": (
            "ComfyUI 侧完成生成或处理后，可回到 MedFusion 主链继续做训练与结果回流。"
        ),
    }


@router.get("/adapter-profiles")
async def get_comfyui_adapter_profiles() -> dict[str, Any]:
    """Return MedFusion-side adapter profiles for ComfyUI workflow alignment."""
    profiles: list[dict[str, Any]] = []
    for blueprint in ADVANCED_BUILDER_BLUEPRINTS:
        if blueprint.status != "compile_ready":
            continue
        components = []
        seen_families: set[str] = set()
        family_chain: list[dict[str, Any]] = []

        for component_id in blueprint.components:
            component = _COMPONENTS_BY_ID.get(component_id)
            if component is None:
                continue
            components.append(
                {
                    "component_id": component.id,
                    "label": component.label,
                    "family": component.family,
                    "family_label": ADVANCED_BUILDER_FAMILY_LABELS[component.family],
                    "status": component.status,
                }
            )
            if component.family not in seen_families:
                seen_families.add(component.family)
                family_chain.append(
                    {
                        "family": component.family,
                        "label": ADVANCED_BUILDER_FAMILY_LABELS[component.family],
                    }
                )

        profiles.append(
            {
                "id": blueprint.id,
                "label": blueprint.label,
                "description": blueprint.description,
                "blueprint_id": blueprint.id,
                "target_canvas_route": f"/config/advanced/canvas?blueprint={blueprint.id}",
                "components": components,
                "family_chain": family_chain,
                "default_import_prefill": _DEFAULT_PREFILL_BY_BLUEPRINT.get(
                    blueprint.id,
                    _DEFAULT_PREFILL_BY_BLUEPRINT["quickstart_multimodal"],
                ),
            }
        )

    return {
        "mode": "adapter_preview",
        "source_boundary": (
            "ComfyUI 作为外部流程编排与交互层；MedFusion runtime 仍是训练执行与结果合同真源。"
        ),
        "recommended_steps": [
            "选择适配档案并进入对应 MedFusion 组件骨架",
            "在高级模式画布完成编译与约束校验",
            "按需回流到结果后台导入真实 run 产物",
        ],
        "profiles": profiles,
    }
