"""MedFusion Web UI 模块

提供基于 FastAPI 的 Web 界面，用于可视化训练、模型管理和实验追踪。
"""

__version__ = "0.3.0"

from .app import app
from .config import settings

__all__ = ["app", "settings", "__version__"]
