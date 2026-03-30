from pathlib import Path

from med_core.version import __version__
from med_core.web.static_assets import resolve_static_asset_location


def test_resolve_static_asset_location_prefers_bundled(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    bundled_dir = package_root / "static"
    downloaded_dir = tmp_path / "data" / "web-ui" / __version__ / "static"

    bundled_dir.mkdir(parents=True)
    downloaded_dir.mkdir(parents=True)
    (bundled_dir / "index.html").write_text("bundled", encoding="utf-8")
    (downloaded_dir / "index.html").write_text("downloaded", encoding="utf-8")

    location = resolve_static_asset_location(
        package_root=package_root,
        data_dir=tmp_path / "data",
        version=__version__,
    )

    assert location is not None
    assert location.directory == bundled_dir
    assert location.source == "bundled"


def test_resolve_static_asset_location_falls_back_to_downloaded(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    downloaded_dir = tmp_path / "data" / "web-ui" / __version__ / "static"

    downloaded_dir.mkdir(parents=True)
    (downloaded_dir / "index.html").write_text("downloaded", encoding="utf-8")

    location = resolve_static_asset_location(
        package_root=package_root,
        data_dir=tmp_path / "data",
        version=__version__,
    )

    assert location is not None
    assert location.directory == downloaded_dir
    assert location.source == "downloaded"


def test_resolve_static_asset_location_returns_none_when_missing(tmp_path: Path) -> None:
    location = resolve_static_asset_location(
        package_root=tmp_path / "package",
        data_dir=tmp_path / "data",
        version=__version__,
    )

    assert location is None
