from pathlib import Path

import numpy as np
import yaml
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from med_core.cli.build_results import build_results
from med_core.cli.doctor import validate_config
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


def test_three_phase_mainline_e2e(tmp_path: Path) -> None:
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
                "experiment_name": "three_phase_mainline_smoke",
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

    validate_config(["--config", str(config_path)])
    train(["--config", str(config_path)])
    build_results(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(output_dir / "checkpoints" / "best.pth"),
        ]
    )

    assert (output_dir / "checkpoints" / "best.pth").exists()
    assert (output_dir / "logs" / "history.json").exists()
    assert (output_dir / "metrics" / "metrics.json").exists()
    assert (output_dir / "metrics" / "validation.json").exists()
    assert (output_dir / "reports" / "summary.json").exists()
    assert (output_dir / "reports" / "report.md").exists()
