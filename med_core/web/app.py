"""FastAPI 应用主文件"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import close_db, init_db

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("正在初始化 MedFusion Web UI...")

    # 初始化目录
    settings.initialize_directories()
    logger.info(f"数据目录: {settings.data_dir}")

    # 初始化数据库
    init_db()
    logger.info("数据库初始化完成")

    logger.info(f"MedFusion Web UI v{settings.version} 启动成功")
    logger.info(f"访问地址: http://{settings.host}:{settings.port}")

    yield

    # 关闭时
    logger.info("正在关闭 MedFusion Web UI...")
    close_db()
    logger.info("MedFusion Web UI 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="MedFusion 医学多模态深度学习框架 Web 界面",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 版本检查中间件
@app.middleware("http")
async def version_check_middleware(request: Request, call_next):
    """检查前后端版本兼容性"""
    if request.url.path.startswith("/api/"):
        client_version = request.headers.get("X-Client-Version")
        if client_version and client_version != settings.version:
            logger.warning(
                f"版本不匹配: 客户端 {client_version}, 服务端 {settings.version}"
            )

    response = await call_next(request)
    response.headers["X-Server-Version"] = settings.version
    return response


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "内部服务器错误",
            "message": str(exc) if settings.debug else "请联系管理员",
        },
    )


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": settings.version,
        "data_dir": str(settings.data_dir),
    }


# API 路由
from .api import datasets, experiments, models, system, training

app.include_router(system.router, prefix="/api/system", tags=["系统"])
app.include_router(training.router, prefix="/api/training", tags=["训练"])
app.include_router(models.router, prefix="/api/models", tags=["模型"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["数据集"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["实验"])


# 静态文件服务（前端）
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    logger.info(f"前端静态文件: {static_dir}")
else:
    logger.warning(f"前端静态文件不存在: {static_dir}")

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "MedFusion Web UI",
            "version": settings.version,
            "status": "前端资源未安装",
            "hint": "请先构建前端: cd web/frontend && npm run build && cp -r dist/* ../../med_core/web/static/",
        }
