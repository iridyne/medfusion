"""系统 API"""
import platform

import psutil
from fastapi import APIRouter

router = APIRouter()


@router.get("/info")
async def get_system_info():
    """获取系统信息"""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
    }


@router.get("/resources")
async def get_system_resources():
    """获取系统资源使用情况"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    # 尝试获取 GPU 信息
    gpu_info = []
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

            gpu_info.append({
                "id": i,
                "name": name,
                "memory_used": memory_info.used,
                "memory_total": memory_info.total,
                "memory_percent": memory_info.used / memory_info.total * 100,
                "utilization": utilization.gpu,
            })

        pynvml.nvmlShutdown()
    except Exception:
        gpu_info = []

    return {
        "cpu": {
            "percent": cpu_percent,
            "count": psutil.cpu_count(),
        },
        "memory": {
            "used": memory.used,
            "total": memory.total,
            "percent": memory.percent,
        },
        "gpu": gpu_info,
    }
