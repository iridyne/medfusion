from pathlib import Path


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_release_smoke_matrix_declares_three_modes_and_core_checks() -> None:
    text = _read_text("docs/contents/playbooks/release-smoke-matrix.md")
    assert "本机浏览器模式" in text
    assert "私有服务器 / 自建部署模式" in text
    assert "托管云模式" in text
    assert "FastAPI BFF + Python worker" in text
    assert "bash test/smoke.sh" in text


def test_playbook_and_readme_link_to_release_smoke_matrix() -> None:
    playbooks = _read_text("docs/contents/playbooks/README.md")
    docs_readme = _read_text("docs/README.md")
    root_readme = _read_text("README.md")

    assert "release-smoke-matrix.md" in playbooks
    assert "release-smoke-matrix.md" in docs_readme
    assert "release-smoke-matrix.md" in root_readme


def test_production_doc_references_release_smoke_matrix() -> None:
    text = _read_text("docs/contents/tutorials/deployment/production.md")
    assert "release-smoke-matrix.md" in text
