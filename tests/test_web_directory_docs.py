"""Guards for the curated web directory documentation layout."""

from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_web_readme_is_single_entrypoint_for_current_layout() -> None:
    text = _read_text("web/README.md")

    assert "唯一推荐入口" in text
    assert "uv run medfusion start" in text
    assert "web/frontend/" in text
    assert "med_core/web/" in text
    assert "web/frontend/dist/" in text
    assert "web/frontend/node_modules/" in text
    assert ".env.local" in text


def test_legacy_web_guides_are_removed() -> None:
    legacy_paths = [
        "web/QUICKSTART.md",
        "web/WEB_UI_GUIDE.md",
        "web/CLI_GUIDE.md",
        "web/docs/README.md",
        "web/docs/CLI_VS_SHELL.md",
        "web/test_backend.py",
        "web/test_optimizations.py",
    ]

    for path in legacy_paths:
        assert not Path(path).exists(), path


def test_deprecated_backend_note_points_to_current_docs() -> None:
    text = _read_text("web/backend/DEPRECATED.md")

    assert "历史占位目录" in text
    assert "docs/contents/getting-started/web-ui.md" in text
    assert "web/README.md" in text


def test_beginner_web_guide_points_to_curated_web_readme() -> None:
    text = _read_text("docs/contents/getting-started/web-ui.md")

    assert "../../../web/README.md" in text


def test_web_readme_points_to_real_test_entrypoints() -> None:
    text = _read_text("web/README.md")

    assert "当前真实的 Web 测试入口" in text
    assert "tests/test_web_api_minimal.py" in text
    assert "tests/test_web_training_controls.py" in text
    assert "tests/test_workflow_api.py" in text
