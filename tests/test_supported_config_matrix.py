"""Guards for the MVP-supported YAML/config matrix."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_text(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_starter_readme_declares_official_supported_yaml_set() -> None:
    text = _read_text("configs/starter/README.md")

    assert "官方支持矩阵" in text
    assert "quickstart.yaml" in text
    assert "default.yaml" in text
    assert "YAML 主链" in text


def test_public_dataset_readme_declares_supported_quickstart_profiles() -> None:
    text = _read_text("configs/public_datasets/README.md")

    assert "官方支持矩阵" in text
    assert "pathmnist_quickstart.yaml" in text
    assert "breastmnist_quickstart.yaml" in text
    assert "uci_heart_disease_quickstart.yaml" in text
    assert "第一次成功" in text
