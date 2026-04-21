"""Web 配置管理"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from med_core.version import __version__


class Settings(BaseSettings):
    """Web 应用配置"""

    model_config = SettingsConfigDict(
        env_prefix="MEDFUSION_",
        case_sensitive=False,
    )

    # 应用信息
    app_name: str = "MedFusion Web UI"
    version: str = __version__
    debug: bool = False

    # 服务器配置
    host: str = "127.0.0.1"
    port: int = 8000

    # 数据目录
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".medfusion")

    # 数据库配置
    database_url: str | None = None

    # Redis 配置（可选）
    redis_url: str | None = None

    # 认证配置
    auth_enabled: bool = False
    auth_token: str | None = None
    secret_key: str = "change-this-in-production"

    # 实验功能开关
    enable_experimental_workflow: bool = True
    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_probe_timeout_sec: float = 3.0

    # 文件上传配置
    max_upload_size: int = 500 * 1024 * 1024  # 500MB
    allowed_extensions: set = {
        ".jpg",
        ".jpeg",
        ".png",
        ".dcm",
        ".nii",
        ".nii.gz",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".pth",
        ".pt",
        ".onnx",
    }

    # 日志配置
    log_level: str = "INFO"
    log_file: Path | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # 设置默认数据库 URL
        if self.database_url is None:
            db_path = self.data_dir / "medfusion.db"
            self.database_url = f"sqlite:///{db_path}"

        # 设置默认日志文件
        if self.log_file is None:
            self.log_file = self.data_dir / "logs" / "web.log"

    def initialize_directories(self) -> None:
        """初始化数据目录"""
        directories = [
            self.data_dir,
            self.data_dir / "models",
            self.data_dir / "model-catalog",
            self.data_dir / "model-catalog" / "custom-models",
            self.data_dir / "model-catalog" / "custom-models" / ".trash",
            self.data_dir / "experiments",
            self.data_dir / "datasets",
            self.data_dir / "logs",
            self.data_dir / "settings",
            self.data_dir / "uploads",
            self.data_dir / "web-ui",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# 全局配置实例
settings = Settings()
