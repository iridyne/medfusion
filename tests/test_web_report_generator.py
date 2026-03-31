from pathlib import Path

from med_core.web.report_generator import ReportGenerator


def test_web_report_generator_uses_clinician_friendly_visualization_titles(
    tmp_path: Path,
) -> None:
    plots_dir = tmp_path / "artifacts" / "visualizations"
    plots_dir.mkdir(parents=True, exist_ok=True)

    roc_path = plots_dir / "roc_curve.png"
    confusion_path = plots_dir / "confusion_matrix.png"
    feature_bar_path = plots_dir / "feature_importance_bar.png"
    for path in (roc_path, confusion_path, feature_bar_path):
        path.write_bytes(b"fake")

    generator = ReportGenerator(output_dir=tmp_path)
    assets = generator._resolve_visualization_assets(
        {
            "name": "demo-exp",
            "artifact_paths": {
                "roc_curve_plot_path": str(roc_path),
                "confusion_matrix_plot_path": str(confusion_path),
                "feature_importance_bar_plot_path": str(feature_bar_path),
            },
        }
    )

    titles = [title for title, _path in assets]
    assert "demo-exp - ROC 曲线（区分能力）" in titles
    assert "demo-exp - 混淆矩阵（阳性/阴性判别情况）" in titles
    assert "demo-exp - 关键影响因素条形图" in titles
