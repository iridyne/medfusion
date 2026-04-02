from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

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
    assert "bash test/smoke.sh" in result.stdout
    assert "scripts/smoke_test.py" not in result.stdout


def test_full_regression_ci_mode_uses_shell_smoke_entrypoint() -> None:
    content = (REPO_ROOT / "scripts" / "full_regression.sh").read_text(encoding="utf-8")

    assert "run bash test/smoke.sh" in content
    assert "uv run python scripts/smoke_test.py" not in content


def test_full_regression_full_mode_is_self_contained() -> None:
    content = (REPO_ROOT / "scripts" / "full_regression.sh").read_text(encoding="utf-8")

    assert "run_full_validation()" in content
    assert "bash scripts/local_ci_test.sh" not in content


def test_github_ci_workflow_uses_shell_smoke_entrypoint() -> None:
    content = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "bash test/smoke.sh" in content
    assert "python scripts/smoke_test.py" not in content


def test_verify_ci_fixes_checks_shell_smoke_entrypoint_in_workflow() -> None:
    content = (REPO_ROOT / "scripts" / "verify_ci_fixes.py").read_text(
        encoding="utf-8"
    )

    assert "bash test/smoke.sh" in content
    assert "CI smoke 入口已统一到 test/smoke.sh" in content
    assert "历史 scripts/smoke_test.py 入口" in content


def test_legacy_smoke_test_script_is_relocated_to_dev_diagnostics() -> None:
    legacy_path = REPO_ROOT / "scripts" / "smoke_test.py"
    diagnostic_path = REPO_ROOT / "scripts" / "dev" / "model_stack_diagnostic.py"

    assert not legacy_path.exists()
    assert diagnostic_path.exists()

    content = diagnostic_path.read_text(encoding="utf-8")

    assert "Model Stack Diagnostic for MedFusion" in content
    assert "developer-only diagnostic script" in content
    assert "scripts/dev/model_stack_diagnostic.py" in content
    assert "scripts/smoke_test.py" not in content


@pytest.mark.parametrize(
    ("relative_path", "expected_mode"),
    [
        ("scripts/local_ci_test.sh", "--full"),
        ("scripts/test_ci_locally.sh", "--quick"),
        ("scripts/quick_ci_test.py", "--quick"),
    ],
)
def test_legacy_validation_entrypoints_delegate_to_full_regression(
    relative_path: str, expected_mode: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert "scripts/full_regression.sh" in content
    assert expected_mode in content


@pytest.mark.parametrize(
    ("relative_path", "expected_reference"),
    [
        ("scripts/full_regression.sh", "test/smoke.sh"),
        ("scripts/ci_diagnostic.py", "test/smoke.sh"),
    ],
)
def test_auxiliary_validation_scripts_reference_shell_smoke_entrypoint(
    relative_path: str, expected_reference: str
) -> None:
    content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert expected_reference in content
    assert "scripts/smoke_test.py" not in content


def test_top_level_docs_point_to_script_based_validation_workflow() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for content in (readme, contributing):
        assert "bash scripts/full_regression.sh --quick" in content
        assert "bash scripts/full_regression.sh --ci" in content
        assert "bash scripts/full_regression.sh --full" in content

    assert "tests/test_config_validation.py" in readme
    assert "tests/test_export.py" in readme
