"""系统信息 API"""

import platform
from pathlib import Path
from typing import Any, Literal

import psutil
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from ..application.ui_preferences import UIPreferencesStore
from ..auth import is_jwt_runtime_available
from ..config import settings

router = APIRouter()


class UIPreferencesRequest(BaseModel):
    history_display_mode: Literal["friendly", "technical"] = "friendly"
    language: Literal["zh", "en"] = "zh"
    theme_mode: Literal["light", "dark", "auto"] = "auto"


def _ui_preferences_store() -> UIPreferencesStore:
    return UIPreferencesStore(settings.data_dir / "settings" / "ui-preferences.json")


@router.get("/features")
async def get_feature_status() -> dict[str, Any]:
    """Expose stable and experimental feature boundaries for the current MVP."""
    return {
        "stable_paths": [
            "/workbench",
            "/datasets",
            "/config",
            "/training",
            "/models",
            "/system",
        ],
        "recommended_primary_flow": [
            "workbench",
            "run_wizard",
            "training_monitor",
            "model_library",
        ],
        "deployment_modes": [
            {
                "id": "local_browser",
                "label": "本机浏览器模式",
                "status": "recommended_now",
                "frontend": "React build served by FastAPI",
                "api_bff": "FastAPI",
                "worker": "Local Python subprocess worker",
                "metadata_store": "SQLite",
                "artifact_store": "Local filesystem",
                "same_capabilities_as_runtime": True,
            },
            {
                "id": "private_server",
                "label": "私有服务器 / 自建部署模式",
                "status": "supported_direction",
                "frontend": "Static frontend served separately or via FastAPI",
                "api_bff": "FastAPI",
                "worker": "Independent Python training worker on GPU host",
                "metadata_store": "PostgreSQL",
                "artifact_store": "Object storage or shared filesystem",
                "same_capabilities_as_runtime": True,
            },
            {
                "id": "managed_cloud",
                "label": "托管云模式",
                "status": "future_direction",
                "frontend": "Static frontend on CDN / gateway",
                "api_bff": "FastAPI",
                "worker": "Multiple independent Python workers",
                "metadata_store": "PostgreSQL",
                "artifact_store": "S3 / OSS / MinIO",
                "same_capabilities_as_runtime": True,
            },
        ],
        "advanced_builder": {
            "route": "/config/advanced",
            "canvas_route": "/config/advanced/canvas",
            "status": "preview",
            "ui_exposed": True,
            "default_entry": False,
            "message": (
                "Advanced builder is exposed as a formal-release preview for "
                "component registry, connection constraints, and compile-ready "
                "blueprints. It is not the default entrypoint."
            ),
            "supported_families": [
                "data_input",
                "vision_backbone",
                "tabular_encoder",
                "fusion",
                "head",
                "training_strategy",
            ],
        },
        "comfyui_bridge": {
            "route": "/config/comfyui",
            "adapter_profiles_api": "/api/comfyui/adapter-profiles",
            "status": "preview",
            "ui_exposed": True,
            "default_entry": False,
            "base_url": settings.comfyui_base_url,
            "message": (
                "ComfyUI bridge provides connectivity check and handoff guidance. "
                "It does not replace MedFusion runtime as the execution source of truth."
            ),
        },
        "auth": {
            "enabled": settings.auth_enabled,
            "jwt_runtime_available": is_jwt_runtime_available(),
            "mode": (
                "disabled"
                if not settings.auth_enabled
                else (
                    "static_token"
                    if settings.auth_token
                    else "jwt_password"
                    if settings.auth_password is not None and is_jwt_runtime_available()
                    else "jwt_runtime_unavailable"
                    if settings.auth_password is not None
                    else "jwt_not_configured"
                )
            ),
            "token_endpoint": "/api/auth/token",
            "rbac_roles": ["viewer", "operator", "admin"],
            "read_permission": "viewer+",
            "write_permission": "operator+",
        },
        "training_queue": {
            "backend": settings.training_queue_backend,
            "redis_url_configured": settings.redis_url is not None,
            "queue_name": settings.redis_queue_name,
            "status": (
                "local_default"
                if str(settings.training_queue_backend).strip().lower() != "redis"
                else "redis_configured"
                if settings.redis_url is not None
                else "redis_default_url"
            ),
        },
        "workflow": {
            "enabled": settings.enable_experimental_workflow,
            "status": (
                "preview" if settings.enable_experimental_workflow else "disabled"
            ),
            "ui_exposed": settings.enable_experimental_workflow,
            "message": (
                "Workflow editor is exposed as a constrained preview. "
                "It currently supports single-mainline orchestration that "
                "hands real execution back to the stable training runtime."
            ),
            "recommended_instead": [
                "Use /workflow for constrained graph-to-training preview",
                "Use Run Wizard for the most stable config authoring path",
                "Use training monitor and model library for runtime and results",
            ],
        },
    }


@router.get("/info")
async def get_system_info() -> dict[str, Any]:
    """获取系统信息"""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "data_dir": str(settings.data_dir),
    }


@router.get("/preferences")
async def get_ui_preferences() -> dict[str, Any]:
    return {
        "preferences": _ui_preferences_store().load(),
        "storage": "filesystem",
        "path": str(settings.data_dir / "settings" / "ui-preferences.json"),
        "history_display_scope": "custom_model_history_only",
    }


@router.put("/preferences")
async def update_ui_preferences(request: UIPreferencesRequest) -> dict[str, Any]:
    preferences = _ui_preferences_store().save(request.model_dump())
    return {
        "preferences": preferences,
        "storage": "filesystem",
        "path": str(settings.data_dir / "settings" / "ui-preferences.json"),
        "history_display_scope": "custom_model_history_only",
    }


@router.delete("/preferences")
async def reset_ui_preferences() -> dict[str, Any]:
    preferences = _ui_preferences_store().reset()
    return {
        "preferences": preferences,
        "storage": "filesystem",
        "path": str(settings.data_dir / "settings" / "ui-preferences.json"),
        "history_display_scope": "custom_model_history_only",
    }


@router.get("/version")
async def get_version() -> dict[str, str]:
    """获取版本信息"""
    return {
        "backend": settings.version,
        "api": "v1",
        "frontend_required": settings.version,
    }


@router.get("/resources")
async def get_system_resources() -> dict[str, Any]:
    """获取系统资源使用情况"""
    # CPU 信息
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()

    # 内存信息
    memory = psutil.virtual_memory()

    # GPU 信息
    gpu_info = []
    if torch.cuda.is_available():
        gpu_info = [
            {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory
                / 1024**3,  # GB
                "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
            }
            for i in range(torch.cuda.device_count())
        ]

    return {
        "cpu": {
            "usage_percent": cpu_percent,
            "count": cpu_count,
        },
        "memory": {
            "total": memory.total / 1024**3,  # GB
            "available": memory.available / 1024**3,  # GB
            "used": memory.used / 1024**3,  # GB
            "percent": memory.percent,
        },
        "gpu": gpu_info,
    }


@router.get("/storage")
async def get_storage_info() -> dict[str, Any]:
    """获取存储信息"""
    data_dir = settings.data_dir

    # 计算各目录大小
    def get_dir_size(path: Path) -> float:
        total = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total / 1024**3  # GB

    return {
        "data_dir": str(data_dir),
        "total_size_gb": get_dir_size(data_dir),
        "models_size_gb": get_dir_size(data_dir / "models"),
        "experiments_size_gb": get_dir_size(data_dir / "experiments"),
        "datasets_size_gb": get_dir_size(data_dir / "datasets"),
    }
