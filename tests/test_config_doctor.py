"""Tests for config doctor readiness checks."""

from pathlib import Path

from med_core.configs import analyze_config, load_config, save_config


def test_quickstart_config_doctor_passes() -> None:
    report = analyze_config("configs/starter/quickstart.yaml")

    assert report.ok is True
    assert report.summary["dataset_rows"] > 0
    assert report.summary["estimated_split_counts"]["train"] > 0


def test_config_doctor_reports_missing_column(tmp_path: Path) -> None:
    config = load_config("configs/starter/quickstart.yaml")
    config.data.target_column = "missing_target"

    config_path = tmp_path / "broken.yaml"
    save_config(config, config_path)

    report = analyze_config(config_path)

    assert report.ok is False
    assert any(item.code == "D104" for item in report.errors)
