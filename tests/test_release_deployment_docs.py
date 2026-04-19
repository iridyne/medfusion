from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_production_doc_declares_three_deployment_modes() -> None:
    text = _read_text("docs/contents/tutorials/deployment/production.md")
    assert "本机浏览器模式" in text
    assert "私有服务器 / 自建部署模式" in text
    assert "托管云模式" in text


def test_production_doc_keeps_fastapi_bff_and_python_worker_boundary() -> None:
    text = _read_text("docs/contents/tutorials/deployment/production.md")
    assert "FastAPI" in text
    assert "Python worker" in text
    assert "不引入 Node 后端" in text


def test_quick_reference_links_deployment_guidance() -> None:
    text = _read_text("docs/contents/guides/core/quick-reference.md")
    assert "docs/contents/getting-started/web-ui.md" in text
    assert "docs/contents/getting-started/public-datasets.md" in text
