"""Static frontend asset resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StaticAssetLocation:
    """Resolved frontend asset location."""

    directory: Path
    source: str


def resolve_static_asset_location(
    *,
    package_root: Path,
    data_dir: Path,
    version: str,
) -> StaticAssetLocation | None:
    """Resolve the frontend static directory used by the web server.

    Preference order:
    1. Bundled package assets under ``med_core/web/static``
    2. Downloaded assets under ``<data_dir>/web-ui/<version>/static``
    """
    package_static_dir = package_root / "static"
    if (package_static_dir / "index.html").exists():
        return StaticAssetLocation(directory=package_static_dir, source="bundled")

    downloaded_static_dir = data_dir / "web-ui" / version / "static"
    if (downloaded_static_dir / "index.html").exists():
        return StaticAssetLocation(directory=downloaded_static_dir, source="downloaded")

    return None
