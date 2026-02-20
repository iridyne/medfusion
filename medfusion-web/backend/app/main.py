"""FastAPI 应用主入口"""

import logging
import traceback
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, datasets, models, preprocessing, system, training, workflows
from app.core.config import settings
from app.core.database import init_db

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting MedFusion Web API...")
    init_db()
    logger.info("Database initialized")
    yield
    # 关闭时
    logger.info("Shutting down MedFusion Web API...")


# 创建 FastAPI 应用
app = FastAPI(
    title="MedFusion Web API",
    description="医学深度学习框架 Web 接口",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# 添加 GZip 压缩中间件
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(
        f"Unhandled exception: {exc}\n"
        f"Request: {request.method} {request.url}\n"
        f"Traceback: {traceback.format_exc()}"
    )

    # 开发环境返回详细错误信息
    if settings.DEBUG:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback.format_exc().split("\n"),
            },
        )

    # 生产环境返回通用错误信息
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """404 错误处理"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": f"Path {request.url.path} not found"},
    )


# 注册路由（添加 API 版本控制）
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(
    preprocessing.router, prefix="/api/v1/preprocessing", tags=["preprocessing"]
)
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "MedFusion Web API",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health/live")
async def liveness():
    """存活检查"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    """就绪检查"""
    from app.core.database import SessionLocal

    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not ready", "database": "disconnected"}


@app.get("/health")
async def health_check():
    """健康检查（兼容旧版本）"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
