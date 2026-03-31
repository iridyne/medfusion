from pathlib import Path

import numpy as np
import yaml
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

from med_core.configs.doctor import ConfigDoctor
from med_core.datasets.three_phase_ct import ThreePhaseCTCaseDataset
from med_core.shared.preprocessing.clinical import ClinicalFeaturePreprocessor


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


def _write_three_phase_case(root: Path, case_id: str) -> dict[str, str]:
    phase_dirs: dict[str, str] = {}
    for phase in ("arterial", "portal", "noncontrast"):
        phase_dir = root / case_id / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            _write_slice(
                phase_dir / f"{index:03d}.dcm",
                instance_number=index + 1,
                pixel_array=np.full((8, 8), fill_value=index + 10, dtype=np.int16),
            )
        phase_dirs[phase] = str(phase_dir)
    return phase_dirs


def test_three_phase_dataset_returns_case_tensors(tmp_path: Path) -> None:
    phase_dirs = _write_three_phase_case(tmp_path, "001")

    dataset = ThreePhaseCTCaseDataset.from_records(
        [
            {
                "case_id": "001",
                "arterial_series_dir": phase_dirs["arterial"],
                "portal_series_dir": phase_dirs["portal"],
                "noncontrast_series_dir": phase_dirs["noncontrast"],
                "mvi_binary": 1,
                "clinical_features": [64.0, 0.0, 24.4],
            }
        ],
        target_shape=(4, 8, 8),
    )

    sample = dataset[0]
    assert sample["case_id"] == "001"
    assert sample["label"].item() == 1
    assert sample["arterial"].shape == (1, 4, 8, 8)
    assert sample["portal"].shape == (1, 4, 8, 8)
    assert sample["noncontrast"].shape == (1, 4, 8, 8)
    assert sample["clinical"].shape == (3,)
    assert sample["clinical_missing_mask"].shape == (3,)
    assert sample["clinical_missing_mask"].tolist() == [0.0, 0.0, 0.0]


def test_three_phase_dataset_uses_clinical_preprocessor_and_missing_mask(
    tmp_path: Path,
) -> None:
    phase_dirs = _write_three_phase_case(tmp_path, "001")
    preprocessor = ClinicalFeaturePreprocessor(
        strategy="zero_with_mask",
        normalize=True,
    )
    preprocessor.fit([[60.0, None, 20.0], [68.0, 1.0, 28.0]])

    dataset = ThreePhaseCTCaseDataset.from_records(
        [
            {
                "case_id": "001",
                "arterial_series_dir": phase_dirs["arterial"],
                "portal_series_dir": phase_dirs["portal"],
                "noncontrast_series_dir": phase_dirs["noncontrast"],
                "mvi_binary": 1,
                "clinical_features": [64.0, None, 24.0],
            }
        ],
        target_shape=(4, 8, 8),
        clinical_preprocessor=preprocessor,
    )

    sample = dataset[0]
    assert sample["clinical"].shape == (3,)
    assert sample["clinical_missing_mask"].tolist() == [0.0, 1.0, 0.0]
    assert np.isclose(sample["clinical"][0].item(), 0.0)
    assert sample["clinical"][1].item() == 0.0


def test_three_phase_doctor_validates_manifest_columns_and_phase_dirs(
    tmp_path: Path,
) -> None:
    phase_dirs_001 = _write_three_phase_case(tmp_path, "001")
    phase_dirs_002 = _write_three_phase_case(tmp_path, "002")
    manifest_path = tmp_path / "cases.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "case_id,arterial_series_dir,portal_series_dir,noncontrast_series_dir,mvi_binary,age,sex,bmi",
                f"001,{phase_dirs_001['arterial']},{phase_dirs_001['portal']},{phase_dirs_001['noncontrast']},1,64,0,24.4",
                f"002,{phase_dirs_002['arterial']},{phase_dirs_002['portal']},{phase_dirs_002['noncontrast']},0,58,1,22.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "smurf_mainline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smurf_mvi_demo",
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
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "model": {
                    "model_type": "three_phase_ct_fusion",
                    "num_classes": 2,
                    "phase_feature_dim": 16,
                    "tabular": {"hidden_dims": [16], "output_dim": 8, "dropout": 0.1},
                    "fusion": {"fusion_type": "gated", "hidden_dim": 12, "dropout": 0.1},
                },
                "training": {
                    "num_epochs": 1,
                    "mixed_precision": False,
                    "use_progressive_training": False,
                    "optimizer": {"optimizer": "adam", "learning_rate": 0.0005},
                    "scheduler": {"scheduler": "none"},
                },
                "logging": {"output_dir": str(tmp_path / "outputs")},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report = ConfigDoctor().analyze(config_path)

    assert report.ok is True
    assert report.errors == []
