"""数据库配置和管理"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

# 创建数据库引擎
engine = create_engine(
    settings.database_url or "sqlite:///./med_core.db",
    connect_args={"check_same_thread": False}
    if "sqlite" in (settings.database_url or "")
    else {},
    pool_pre_ping=True,
    echo=settings.debug,
)

# 会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类
Base = declarative_base()


def init_db() -> None:
    """初始化数据库"""
    # SQLite 优化
    if "sqlite" in (settings.database_url or ""):
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA cache_size=-64000"))  # 64MB cache
            conn.commit()
        logger.info("SQLite 优化配置已应用")

    # 创建所有表
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表创建完成")


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


def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（依赖注入）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
