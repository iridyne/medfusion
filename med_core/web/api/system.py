"""系统信息 API"""

import platform
from typing import Any

import psutil
import torch
from fastapi import APIRouter

from ..config import settings

router = APIRouter()


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
        for i in range(torch.cuda.device_count()):
            gpu_info.append(
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory
                    / 1024**3,  # GB
                    "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
                    "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
                }
            )

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
