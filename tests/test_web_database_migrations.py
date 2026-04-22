"""Tests for DB URL normalization and migration runtime fallbacks."""

from med_core.web import migrations
from med_core.web.config import normalize_database_url


def test_normalize_database_url_converts_postgres_short_scheme() -> None:
    normalized = normalize_database_url(
        "postgres://user:pass@127.0.0.1:5432/medfusion",
    )
    assert normalized == "postgresql+psycopg://user:pass@127.0.0.1:5432/medfusion"


def test_normalize_database_url_converts_postgresql_default_scheme() -> None:
    normalized = normalize_database_url(
        "postgresql://user:pass@127.0.0.1:5432/medfusion",
    )
    assert normalized == "postgresql+psycopg://user:pass@127.0.0.1:5432/medfusion"


def test_normalize_database_url_keeps_existing_driver() -> None:
    normalized = normalize_database_url(
        "postgresql+psycopg://user:pass@127.0.0.1:5432/medfusion",
    )
    assert normalized == "postgresql+psycopg://user:pass@127.0.0.1:5432/medfusion"


def test_upgrade_database_returns_false_when_alembic_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(migrations, "_import_alembic", lambda: None)
    assert (
        migrations.upgrade_database(
            database_url="sqlite:///missing-alembic.db",
            revision="head",
        )
        is False
    )


def test_current_revision_returns_none_when_alembic_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(migrations, "_import_alembic", lambda: None)
    assert migrations.current_revision(database_url="sqlite:///missing-alembic.db") is None

