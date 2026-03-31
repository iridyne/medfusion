import json
from pathlib import Path

import numpy as np
import yaml
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from med_core.cli.build_results import build_results
from med_core.cli.train import train


def _write_slice(path: Path, instance_number: int, pixel_array: np.ndarray) -> None:
    file_meta = Dataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = pixel_array.shape
    dataset.InstanceNumber = instance_number
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 1.0
    dataset.ImagePositionPatient = [0.0, 0.0, float(instance_number)]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelData = pixel_array.astype(np.int16).tobytes()
    dataset.save_as(path)


def _write_case(root: Path, case_id: str, offset: int) -> dict[str, str]:
    phase_dirs: dict[str, str] = {}
    for phase in ("arterial", "portal", "noncontrast"):
        phase_dir = root / case_id / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            _write_slice(
                phase_dir / f"{index:03d}.dcm",
                instance_number=index + 1,
                pixel_array=np.full((8, 8), fill_value=offset + index, dtype=np.int16),
            )
        phase_dirs[phase] = str(phase_dir)
    return phase_dirs


def test_three_phase_build_results_runs_from_mainline_config(tmp_path: Path) -> None:
    case_one = _write_case(tmp_path, "001", 10)
    case_two = _write_case(tmp_path, "002", 20)

    manifest_path = tmp_path / "cases.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "case_id,arterial_series_dir,portal_series_dir,noncontrast_series_dir,mvi_binary,age,sex,bmi",
                f"001,{case_one['arterial']},{case_one['portal']},{case_one['noncontrast']},1,64,0,24.4",
                f"002,{case_two['arterial']},{case_two['portal']},{case_two['noncontrast']},0,58,1,22.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "smurf_mainline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smurf_build_results_smoke",
                "device": "cpu",
                "data": {
                    "dataset_type": "three_phase_ct_tabular",
                    "csv_path": str(manifest_path),
                    "target_column": "mvi_binary",
                    "patient_id_column": "case_id",
                    "phase_dir_columns": {
                        "arterial": "arterial_series_dir",
                        "portal": "portal_series_dir",
                        "noncontrast": "noncontrast_series_dir",
                    },
                    "clinical_feature_columns": ["age", "sex", "bmi"],
                    "target_shape": [4, 8, 8],
                    "window_preset": "liver",
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "train_ratio": 0.5,
                    "val_ratio": 0.5,
                    "test_ratio": 0.0,
                },
                "model": {
                    "model_type": "three_phase_ct_fusion",
                    "num_classes": 2,
                    "phase_feature_dim": 16,
                    "share_phase_encoder": False,
                    "use_risk_head": True,
                    "tabular": {"hidden_dims": [16], "output_dim": 8, "dropout": 0.1},
                    "fusion": {"fusion_type": "gated", "hidden_dim": 12, "dropout": 0.1},
                },
                "training": {
                    "num_epochs": 1,
                    "mixed_precision": False,
                    "use_progressive_training": False,
                    "optimizer": {
                        "optimizer": "adam",
                        "learning_rate": 0.0005,
                        "weight_decay": 0.0,
                    },
                    "scheduler": {"scheduler": "none"},
                },
                "logging": {"output_dir": str(output_dir), "use_tensorboard": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    train(["--config", str(config_path)])
    checkpoint_path = output_dir / "checkpoints" / "best.pth"

    build_results(["--config", str(config_path), "--checkpoint", str(checkpoint_path)])

    assert (output_dir / "metrics" / "metrics.json").exists()
    assert (output_dir / "metrics" / "validation.json").exists()
    assert (output_dir / "metrics" / "case_explanations.json").exists()
    assert (output_dir / "reports" / "summary.json").exists()
    assert (output_dir / "reports" / "report.md").exists()


def test_three_phase_build_results_emits_roc_and_shap_artifacts(tmp_path: Path) -> None:
    rows: list[str] = [
        "case_id,arterial_series_dir,portal_series_dir,noncontrast_series_dir,mvi_binary,age,sex,bmi"
    ]
    for index in range(8):
        case_id = f"{index + 1:03d}"
        phase_dirs = _write_case(tmp_path, case_id, 10 + index * 3)
        label = index % 2
        age = 50 + index
        sex = index % 2
        bmi = 21.0 + index * 0.5
        rows.append(
            f"{case_id},{phase_dirs['arterial']},{phase_dirs['portal']},{phase_dirs['noncontrast']},{label},{age},{sex},{bmi}"
        )

    manifest_path = tmp_path / "cases.csv"
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "smurf_mainline_artifacts.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smurf_build_results_artifacts",
                "device": "cpu",
                "data": {
                    "dataset_type": "three_phase_ct_tabular",
                    "csv_path": str(manifest_path),
                    "target_column": "mvi_binary",
                    "patient_id_column": "case_id",
                    "phase_dir_columns": {
                        "arterial": "arterial_series_dir",
                        "portal": "portal_series_dir",
                        "noncontrast": "noncontrast_series_dir",
                    },
                    "clinical_feature_columns": ["age", "sex", "bmi"],
                    "target_shape": [4, 8, 8],
                    "window_preset": "liver",
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "train_ratio": 1.0,
                    "val_ratio": 0.0,
                    "test_ratio": 0.0,
                },
                "model": {
                    "model_type": "three_phase_ct_fusion",
                    "num_classes": 2,
                    "phase_feature_dim": 16,
                    "share_phase_encoder": False,
                    "use_risk_head": True,
                    "tabular": {"hidden_dims": [16], "output_dim": 8, "dropout": 0.1},
                    "fusion": {"fusion_type": "gated", "hidden_dim": 12, "dropout": 0.1},
                },
                "training": {
                    "num_epochs": 1,
                    "mixed_precision": False,
                    "use_progressive_training": False,
                    "optimizer": {
                        "optimizer": "adam",
                        "learning_rate": 0.0005,
                        "weight_decay": 0.0,
                    },
                    "scheduler": {"scheduler": "none"},
                },
                "logging": {"output_dir": str(output_dir), "use_tensorboard": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    train(["--config", str(config_path)])
    checkpoint_path = output_dir / "checkpoints" / "best.pth"

    build_results(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--split",
            "train",
        ]
    )

    assert (output_dir / "artifacts" / "visualizations" / "roc_curve.png").exists()
    assert (
        output_dir / "artifacts" / "visualizations" / "confusion_matrix.png"
    ).exists()
    assert (
        output_dir / "artifacts" / "visualizations" / "shap" / "shap_bar.png"
    ).exists()
    assert (
        output_dir / "artifacts" / "visualizations" / "shap" / "shap_beeswarm.png"
    ).exists()

    summary = json.loads((output_dir / "reports" / "summary.json").read_text())
    artifacts = summary["artifacts"]
    assert artifacts["config_path"].endswith("artifacts/training-config.json")
    assert artifacts["metrics_path"].endswith("metrics/metrics.json")
    assert artifacts["validation_path"].endswith("metrics/validation.json")
    assert artifacts["predictions_path"].endswith("metrics/predictions.json")
    assert artifacts["case_explanations_path"].endswith(
        "metrics/case_explanations.json"
    )
    assert artifacts["phase_importance_path"].endswith(
        "metrics/phase_importance.json"
    )
    assert artifacts["history_path"].endswith("logs/history.json")
    assert artifacts["roc_curve_plot_path"].endswith("artifacts/visualizations/roc_curve.png")
    assert artifacts["confusion_matrix_plot_path"].endswith(
        "artifacts/visualizations/confusion_matrix.png"
    )
    assert artifacts["shap_bar_plot_path"].endswith(
        "artifacts/visualizations/shap/shap_bar.png"
    )
    assert artifacts["shap_beeswarm_plot_path"].endswith(
        "artifacts/visualizations/shap/shap_beeswarm.png"
    )
    assert artifacts["feature_importance_bar_plot_path"].endswith(
        "artifacts/visualizations/shap/shap_bar.png"
    )
    assert artifacts["feature_importance_beeswarm_plot_path"].endswith(
        "artifacts/visualizations/shap/shap_beeswarm.png"
    )
    assert artifacts["feature_importance_path"].endswith("artifacts/shap_summary.json")

    report_text = (output_dir / "reports" / "report.md").read_text(encoding="utf-8")
    assert "## Overview" in report_text
    assert "## Metrics" in report_text
    assert "## Data Summary" in report_text
    assert "## Visual Artifacts" in report_text
    assert "## Feature Importance" in report_text
    assert "## Phase Contribution" in report_text
    assert "## Artifact Paths" in report_text
    assert "- 区分能力（AUC）:" in report_text
    assert "- 总体准确率:" in report_text
    assert "- 阳性病例数:" in report_text
    assert "- 阴性病例数:" in report_text
    assert "- 纳入的临床变量:" in report_text
    assert "- ROC 曲线（区分能力）:" in report_text
    assert "- 混淆矩阵（阳性/阴性判别情况）:" in report_text
    assert "- 三期贡献概览:" in report_text
    assert "- 方法说明: SHAP-style surrogate" in report_text
    assert "- 关键影响因素条形图:" in report_text
    assert "- 关键影响因素散点图:" in report_text
    assert "- Feature Importance Bar:" not in report_text
    assert "- Feature Importance Beeswarm:" not in report_text

    case_explanations = json.loads(
        (output_dir / "metrics" / "case_explanations.json").read_text()
    )
    assert case_explanations["cases"]
    assert "phase_importance" in case_explanations["cases"][0]
