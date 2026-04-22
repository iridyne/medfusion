"""Smoke guards for scripts/release_smoke.py modes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_smoke_docker_dry_run_mode_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/release_smoke.py", "--mode", "docker-dry-run"],
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Docker dry-run checks passed" in result.stdout
