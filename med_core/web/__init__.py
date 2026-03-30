"""MedFusion Web UI 模块

提供基于 FastAPI 的 Web 界面，用于可视化训练、模型管理和实验追踪。
"""

from med_core.version import __version__

__all__ = ["__version__", "app", "settings"]


def __getattr__(name: str):
    if name == "app":
        from .app import app

        return app
    if name == "settings":
        from .config import settings

        return settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
