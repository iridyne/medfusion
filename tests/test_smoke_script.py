from pathlib import Path


OSS_ROOT = Path(__file__).resolve().parents[1]


def _read_text(relative_path: str) -> str:
    return (OSS_ROOT / relative_path).read_text(encoding="utf-8")


def test_smoke_script_exists_and_anchors_repo_root() -> None:
    script_path = OSS_ROOT / "test" / "smoke.sh"

    assert script_path.exists()
    text = _read_text("test/smoke.sh")
    assert "REPO_ROOT=" in text
    assert 'cd "$REPO_ROOT"' in text


def test_smoke_script_runs_breastmnist_quickstart_mainline() -> None:
    text = _read_text("test/smoke.sh")

    assert "medfusion public-datasets prepare medmnist-breastmnist" in text
    assert "configs/public_datasets/breastmnist_quickstart.yaml" in text
    assert "medfusion validate-config" in text
    assert "medfusion train" in text
    assert "medfusion build-results" in text


def test_smoke_script_checks_canonical_artifacts() -> None:
    text = _read_text("test/smoke.sh")

    expected_artifacts = [
        "checkpoints/best.pth",
        "logs/history.json",
        "metrics/metrics.json",
        "metrics/validation.json",
        "reports/summary.json",
        "reports/report.md",
    ]

    for artifact in expected_artifacts:
        assert artifact in text


def test_readme_mentions_smoke_script_entrypoint() -> None:
    text = _read_text("README.md")

    assert "bash test/smoke.sh" in text
