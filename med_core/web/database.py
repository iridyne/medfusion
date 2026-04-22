"""数据库配置和管理"""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite")


DATABASE_URL = settings.database_url or "sqlite:///./medfusion.db"


# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if _is_sqlite_url(DATABASE_URL)
    else {},
    pool_pre_ping=True,
    echo=settings.debug,
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类
Base = declarative_base()


def _ensure_model_registry_loaded() -> None:
    # Ensure all SQLAlchemy model modules are imported so Base.metadata is complete.
    from . import models as _models  # noqa: F401


def _apply_sqlite_pragmas() -> None:
    if not _is_sqlite_url(DATABASE_URL):
        return
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB cache
        conn.commit()
    logger.info("SQLite 优化配置已应用")


def _prepare_database_runtime() -> None:
    settings.initialize_directories()
    _apply_sqlite_pragmas()


def upgrade_schema(revision: str = "head") -> str:
    """Upgrade DB schema and return mode: migrated/create_all_fallback."""
    _prepare_database_runtime()
    _ensure_model_registry_loaded()

    from .migrations import upgrade_database

    migrated = upgrade_database(database_url=DATABASE_URL, revision=revision)
    if migrated:
        return "migrated"

    Base.metadata.create_all(bind=engine)
    logger.warning(
        "数据库迁移未执行（Alembic 不可用），已回退到 Base.metadata.create_all。"
    )
    return "create_all_fallback"


def current_schema_revision() -> str | None:
    """Return current Alembic revision when migration runtime is available."""
    from .migrations import current_revision

    return current_revision(database_url=DATABASE_URL)


def init_db() -> None:
    """初始化数据库"""
    mode = upgrade_schema("head")
    if mode == "migrated":
        logger.info("数据库迁移已完成（head）")
    logger.info("数据库可用")


def close_db() -> None:
    """关闭数据库连接"""
    engine.dispose()
    logger.info("数据库连接已关闭")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """获取数据库会话（上下文管理器）"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def get_db_session() -> AsyncGenerator[Session, None]:
    """获取数据库会话（依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
