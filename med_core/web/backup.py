"""Backup helpers for Web data and DB metadata snapshots."""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Date,
    DateTime,
    Integer,
    Time,
    delete,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql.schema import Column, Table

from .config import settings
from .database import Base, _ensure_model_registry_loaded, engine

BACKUP_SCHEMA_VERSION = "1.0"
DB_SNAPSHOT_SCHEMA_VERSION = "1.0"
DEFAULT_DATA_ROOT = "data"
DEFAULT_DB_SNAPSHOT_PATH = "db/snapshot.json"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _resolve_archive_path(output_path: Path) -> Path:
    lower_name = output_path.name.lower()
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz") or lower_name.endswith(
        ".tar"
    ):
        return output_path
    return output_path.with_name(f"{output_path.name}.tar.gz")


def _database_backend(database_url: str | None) -> str:
    if not database_url:
        return "unknown"
    normalized = database_url.lower()
    if normalized.startswith("sqlite"):
        return "sqlite"
    if normalized.startswith("postgresql") or normalized.startswith("postgres"):
        return "postgresql"
    return "other"


def _model_tables() -> list[Table]:
    _ensure_model_registry_loaded()
    return list(Base.metadata.sorted_tables)


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    return value


def _deserialize_scalar(column: Column[Any], value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict) and "__bytes_base64__" in value:
        return base64.b64decode(value["__bytes_base64__"])

    if isinstance(value, str):
        if isinstance(column.type, DateTime):
            candidate = value.removesuffix("Z")
            with suppress(ValueError):
                return datetime.fromisoformat(candidate)
        if isinstance(column.type, Date):
            with suppress(ValueError):
                return date.fromisoformat(value)
        if isinstance(column.type, Time):
            with suppress(ValueError):
                return time.fromisoformat(value)

    return value


def export_database_snapshot(*, db_engine: Engine | None = None) -> dict[str, Any]:
    """Export ORM-managed tables into a JSON-serializable snapshot."""
    target_engine = db_engine or engine
    tables = _model_tables()

    payload_tables: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    table_order = [table.name for table in tables]

    with target_engine.connect() as conn:
        existing_tables = set(inspect(conn).get_table_names())
        for table in tables:
            if table.name not in existing_tables:
                payload_tables[table.name] = []
                row_counts[table.name] = 0
                continue
            rows = conn.execute(select(table)).mappings().all()
            serialized_rows = []
            for row in rows:
                serialized_rows.append(
                    {
                        column.name: _serialize_scalar(row.get(column.name))
                        for column in table.columns
                    }
                )
            payload_tables[table.name] = serialized_rows
            row_counts[table.name] = len(serialized_rows)

        alembic_revision = None
        with suppress(Exception):
            alembic_revision = conn.execute(
                text("SELECT version_num FROM alembic_version LIMIT 1"),
            ).scalar_one_or_none()

    return {
        "schema_version": DB_SNAPSHOT_SCHEMA_VERSION,
        "exported_at": _now_iso(),
        "table_order": table_order,
        "row_counts": row_counts,
        "tables": payload_tables,
        "alembic_revision": alembic_revision,
    }


def _coerce_row(table: Table, raw_row: Mapping[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for column in table.columns:
        if column.name not in raw_row:
            continue
        coerced[column.name] = _deserialize_scalar(column, raw_row[column.name])
    return coerced


def _reset_postgres_sequences(conn: Connection, table_order: list[Table]) -> None:
    if conn.dialect.name != "postgresql":
        return

    for table in table_order:
        for column in table.primary_key.columns:
            if not isinstance(column.type, Integer):
                continue
            max_value = conn.execute(
                select(func.max(column)).select_from(table),
            ).scalar_one_or_none()
            max_int = int(max_value or 0)
            conn.execute(
                text(
                    "SELECT setval("
                    "pg_get_serial_sequence(:table_name, :column_name), "
                    ":value, :is_called"
                    ")",
                ),
                {
                    "table_name": table.name,
                    "column_name": column.name,
                    "value": max(max_int, 1),
                    "is_called": max_int > 0,
                },
            )


def restore_database_snapshot(
    snapshot: Mapping[str, Any],
    *,
    truncate_existing: bool = True,
    db_engine: Engine | None = None,
) -> dict[str, int]:
    """Restore ORM-managed tables from a snapshot payload."""
    target_engine = db_engine or engine
    tables = _model_tables()
    table_by_name = {table.name: table for table in tables}

    table_payload = snapshot.get("tables")
    if not isinstance(table_payload, Mapping):
        raise ValueError("Invalid database snapshot: missing tables mapping.")

    raw_order = snapshot.get("table_order")
    if isinstance(raw_order, list):
        ordered_names = [name for name in raw_order if name in table_by_name]
    else:
        ordered_names = [table.name for table in tables]

    ordered_tables = [table_by_name[name] for name in ordered_names]
    restored_counts: dict[str, int] = {}

    with target_engine.begin() as conn:
        if truncate_existing:
            for table in reversed(ordered_tables):
                conn.execute(delete(table))

        for table in ordered_tables:
            raw_rows = table_payload.get(table.name, [])
            if not isinstance(raw_rows, list):
                raise ValueError(f"Invalid database snapshot rows for table: {table.name}")

            prepared_rows = []
            for raw_row in raw_rows:
                if not isinstance(raw_row, Mapping):
                    raise ValueError(
                        f"Invalid database snapshot row for table: {table.name}",
                    )
                prepared_rows.append(_coerce_row(table, raw_row))

            if prepared_rows:
                conn.execute(insert(table), prepared_rows)
            restored_counts[table.name] = len(prepared_rows)

        _reset_postgres_sequences(conn, ordered_tables)

    return restored_counts


def create_backup_archive(
    output_path: Path,
    *,
    include_db_snapshot: bool = True,
    source_data_dir: Path | None = None,
    db_engine: Engine | None = None,
) -> Path:
    """Create backup archive and return absolute archive path."""
    data_dir = (source_data_dir or settings.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    archive_path = _resolve_archive_path(output_path.expanduser().resolve())
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    db_snapshot = None
    if include_db_snapshot:
        db_snapshot = export_database_snapshot(db_engine=db_engine)

    manifest: dict[str, Any] = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "medfusion_version": settings.version,
        "data_root": DEFAULT_DATA_ROOT,
        "database": {
            "backend": _database_backend(settings.database_url),
            "snapshot_included": db_snapshot is not None,
            "snapshot_path": DEFAULT_DB_SNAPSHOT_PATH if db_snapshot is not None else None,
            "snapshot_row_counts": db_snapshot.get("row_counts", {})
            if db_snapshot is not None
            else {},
            "alembic_revision": db_snapshot.get("alembic_revision")
            if db_snapshot is not None
            else None,
        },
    }

    with tempfile.TemporaryDirectory(prefix="medfusion-backup-payload-") as temp_dir:
        payload_root = Path(temp_dir)
        staged_root = payload_root / "payload"
        staged_root.mkdir(parents=True, exist_ok=True)

        (staged_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if db_snapshot is not None:
            db_root = staged_root / "db"
            db_root.mkdir(parents=True, exist_ok=True)
            (db_root / "snapshot.json").write_text(
                json.dumps(db_snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        shutil.copytree(
            data_dir,
            staged_root / DEFAULT_DATA_ROOT,
            dirs_exist_ok=True,
        )

        import tarfile

        with tarfile.open(archive_path, "w:gz") as tar:
            for entry in staged_root.iterdir():
                tar.add(entry, arcname=entry.name)

    return archive_path


def discover_backup_contents(
    extracted_root: Path,
) -> tuple[Path, dict[str, Any] | None, Path | None]:
    """Locate data root and optional DB snapshot in extracted backup contents."""
    manifest_path = extracted_root / "manifest.json"
    manifest: dict[str, Any] | None = None
    db_snapshot_path: Path | None = None

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_root_name = str(manifest.get("data_root") or DEFAULT_DATA_ROOT)
        data_root = extracted_root / data_root_name
        if not data_root.exists() or not data_root.is_dir():
            raise ValueError(
                f"Invalid backup manifest: data root '{data_root_name}' not found.",
            )

        database_meta = manifest.get("database")
        if isinstance(database_meta, Mapping):
            snapshot_path_raw = database_meta.get("snapshot_path")
            if isinstance(snapshot_path_raw, str) and snapshot_path_raw:
                candidate = extracted_root / snapshot_path_raw
                if candidate.exists() and candidate.is_file():
                    db_snapshot_path = candidate
        return data_root, manifest, db_snapshot_path

    # 兼容历史 backup 格式：根目录直接是数据内容。
    extracted_dirs = [item for item in extracted_root.iterdir() if item.is_dir()]
    extracted_files = [item for item in extracted_root.iterdir() if item.is_file()]
    known_data_children = {
        "models",
        "experiments",
        "datasets",
        "logs",
        "uploads",
        "settings",
        "web-ui",
        "model-catalog",
        "checkpoints",
        "outputs",
    }
    if (
        len(extracted_dirs) == 1
        and not extracted_files
        and extracted_dirs[0].name not in known_data_children
    ):
        return extracted_dirs[0], None, None
    return extracted_root, None, None


def load_database_snapshot(snapshot_path: Path) -> dict[str, Any]:
    """Load database snapshot JSON from backup archive payload."""
    return json.loads(snapshot_path.read_text(encoding="utf-8"))
