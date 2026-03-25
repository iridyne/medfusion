from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_regression_entrypoint_exists_and_is_executable() -> None:
    script = REPO_ROOT / "scripts" / "full_regression.sh"

    assert script.exists()
    assert script.stat().st_mode & stat.S_IXUSR


def test_full_regression_help_documents_supported_modes() -> None:
    script = REPO_ROOT / "scripts" / "full_regression.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "TERM": "dumb"},
    )

    assert result.returncode == 0
    assert "quick" in result.stdout
    assert "ci" in result.stdout
    assert "full" in result.stdout
    assert "tests/test_config_validation.py" in result.stdout
    assert "tests/test_export.py" in result.stdout


def test_top_level_docs_point_to_script_based_validation_workflow() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for content in (readme, contributing):
        assert "bash scripts/full_regression.sh --quick" in content
        assert "bash scripts/full_regression.sh --ci" in content
        assert "bash scripts/full_regression.sh --full" in content

    assert "tests/test_config_validation.py" in readme
    assert "tests/test_export.py" in readme
