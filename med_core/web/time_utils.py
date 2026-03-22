"""Time helpers for the web layer."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return a naive UTC datetime compatible with current SQLAlchemy columns."""
    return datetime.now(UTC).replace(tzinfo=None)
