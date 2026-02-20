"""应用配置"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    APP_NAME: str = "MedFusion Web"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS 配置
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///./medfusion.db"
    
    # Redis 配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery 配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # 文件存储
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    LOGS_DIR: str = "./logs"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
