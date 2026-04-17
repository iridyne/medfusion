import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

train_module = importlib.import_module("med_core.cli.train")

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


def test_three_phase_train_runs_from_mainline_config(
    tmp_path: Path,
) -> None:
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
                "experiment_name": "smurf_train_smoke",
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
                    "doctor_interest": {
                        "enabled": True,
                        "hidden_channels": 8,
                        "temperature": 6.0,
                    },
                    "topk_focus": {
                        "enabled": False,
                        "k": 3,
                        "patch_size": [2, 2, 2],
                        "projection_dim": 8,
                    },
                    "share_phase_encoder": False,
                    "use_risk_head": True,
                    "tabular": {"hidden_dims": [16], "output_dim": 8, "dropout": 0.1},
                    "fusion": {"fusion_type": "gated", "hidden_dim": 12, "dropout": 0.1},
                },
                "training": {
                    "num_epochs": 1,
                    "doctor_interest_loss": {
                        "cam_align_weight": 0.05,
                        "consistency_weight": 0.02,
                        "sparse_weight": 0.01,
                        "diverse_weight": 0.01,
                        "body_prior_weight": 0.02,
                    },
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

    history = json.loads((output_dir / "logs" / "history.json").read_text())
    assert "train_doctor_interest_loss" in history["entries"][0]
    assert "val_doctor_interest_loss" in history["entries"][0]
    assert history["entries"][0]["train_doctor_interest_loss"] > 0
    assert history["entries"][0]["val_doctor_interest_loss"] > 0
    assert (output_dir / "checkpoints" / "best.pth").exists()
    assert (output_dir / "logs" / "history.json").exists()
    assert (output_dir / "artifacts" / "clinical_preprocessing.json").exists()


def test_three_phase_doctor_interest_loss_is_positive_without_topk() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(
            doctor_interest=SimpleNamespace(enabled=True),
            topk_focus=SimpleNamespace(enabled=False),
        ),
        training=SimpleNamespace(
            doctor_interest_loss=SimpleNamespace(
                cam_align_weight=0.05,
                consistency_weight=0.02,
                sparse_weight=0.01,
                diverse_weight=0.01,
                body_prior_weight=0.02,
            )
        ),
    )
    prob_map = torch.softmax(torch.randn(1, 64), dim=1).view(1, 1, 4, 4, 4)
    outputs = {
        "doctor_interest_maps": {
            phase: {
                "prob_map": prob_map,
                "score_map": prob_map,
            }
            for phase in ("arterial", "portal", "noncontrast")
        }
    }
    normalized_like_volume = torch.zeros((1, 1, 4, 4, 4))
    normalized_like_volume[:, :, :, 1:3, 1:3] = 0.5
    batch = {
        "arterial": normalized_like_volume,
        "portal": normalized_like_volume,
        "noncontrast": normalized_like_volume,
    }

    total_loss, components = train_module._compute_three_phase_doctor_interest_loss(
        config=config,
        phase_inputs={
            "arterial": batch["arterial"],
            "portal": batch["portal"],
            "noncontrast": batch["noncontrast"],
        },
        outputs=outputs,
        device=torch.device("cpu"),
    )

    assert float(total_loss.item()) > 0
    assert float(components["consistency"].item()) > 0
    assert float(components["body_prior"].item()) > 0
    assert any(float(value.item()) > 0 for value in components.values())


def test_three_phase_body_prior_threshold_switches_for_normalized_inputs() -> None:
    normalized_volume = torch.zeros((1, 1, 4, 4, 4))
    normalized_volume[:, :, :, 1:3, 1:3] = 0.5
    hu_volume = torch.full((1, 1, 4, 4, 4), 40.0)

    assert train_module._resolve_body_prior_threshold(normalized_volume) == 0.1
    assert train_module._resolve_body_prior_threshold(hu_volume) == -200.0

    body_prior = train_module.build_body_mask_prior(
        normalized_volume,
        threshold_hu=train_module._resolve_body_prior_threshold(normalized_volume),
        border_ratio=0.1,
    )
    assert float(body_prior.mean().item()) > 0


def test_three_phase_doctor_interest_loss_single_phase_matches_manual_result() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(
            doctor_interest=SimpleNamespace(enabled=True),
            topk_focus=SimpleNamespace(enabled=False),
        ),
        training=SimpleNamespace(
            doctor_interest_loss=SimpleNamespace(
                cam_align_weight=0.05,
                consistency_weight=0.02,
                sparse_weight=0.01,
                diverse_weight=0.01,
                body_prior_weight=0.02,
            )
        ),
    )
    prob_map = torch.softmax(torch.randn(1, 64), dim=1).view(1, 1, 4, 4, 4)
    single_phase_outputs = {
        "doctor_interest_maps": {
            "arterial": {
                "prob_map": prob_map,
                "score_map": prob_map,
            }
        }
    }
    normalized_like_volume = torch.zeros((1, 1, 4, 4, 4))
    normalized_like_volume[:, :, :, 1:3, 1:3] = 0.5
    phase_inputs = {"arterial": normalized_like_volume}

    manual_body_prior = train_module.build_body_mask_prior(
        normalized_like_volume,
        threshold_hu=train_module._resolve_body_prior_threshold(normalized_like_volume),
        border_ratio=0.1,
    )
    manual_loss = train_module.compute_doctor_interest_losses(
        prob_map=prob_map,
        teacher_map=prob_map,
        augmented_prob_map=0.5 * (prob_map + torch.roll(prob_map, shifts=1, dims=-1)),
        topk_centers=torch.empty((1, 0, 3), dtype=torch.long),
        body_prior=manual_body_prior,
        cam_align_weight=0.05,
        consistency_weight=0.02,
        sparse_weight=0.01,
        diverse_weight=0.01,
        body_prior_weight=0.02,
    )

    total_loss, components = train_module._compute_three_phase_doctor_interest_loss(
        config=config,
        phase_inputs=phase_inputs,
        outputs=single_phase_outputs,
        device=torch.device("cpu"),
    )

    assert torch.isclose(total_loss, manual_loss["total"])
    assert torch.isclose(components["consistency"], manual_loss["components"]["consistency"])
    assert torch.isclose(components["body_prior"], manual_loss["components"]["body_prior"])
