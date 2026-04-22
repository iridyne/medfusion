"""Programmatic Alembic migration helpers for MedFusion Web metadata DB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCRIPT_LOCATION = Path(__file__).resolve().parent / "db_migrations"


def _import_alembic() -> tuple[Any, Any, Any] | None:
    try:
        from alembic import command
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
    except ImportError:
        return None
    return command, Config, MigrationContext


def is_migration_runtime_available() -> bool:
    """Return whether Alembic runtime is available in current environment."""
    return _import_alembic() is not None


def _build_config(*, database_url: str) -> tuple[Any, Any]:
    imports = _import_alembic()
    if imports is None:
        raise RuntimeError("Alembic runtime is unavailable")
    command, Config, _migration_context = imports
    if not SCRIPT_LOCATION.exists():
        raise FileNotFoundError(f"Alembic script location not found: {SCRIPT_LOCATION}")

    config = Config()
    config.set_main_option("script_location", str(SCRIPT_LOCATION))
    config.set_main_option("sqlalchemy.url", database_url)
    return command, config


def upgrade_database(*, database_url: str, revision: str = "head") -> bool:
    """Upgrade database schema to target revision.

    Returns:
        ``True`` when Alembic migration executed; ``False`` when Alembic is not
        available and caller should decide whether to fallback to ``create_all``.
    """
    if not is_migration_runtime_available():
        logger.warning(
            "Alembic runtime is not installed; migration step skipped and caller may fallback."
        )
        return False

    command, config = _build_config(database_url=database_url)
    command.upgrade(config, revision)
    return True


def current_revision(*, database_url: str) -> str | None:
    """Read current Alembic revision from target database."""
    imports = _import_alembic()
    if imports is None:
        return None
    _command, _Config, MigrationContext = imports

    from sqlalchemy import create_engine

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return context.get_current_revision()
    finally:
        engine.dispose()

