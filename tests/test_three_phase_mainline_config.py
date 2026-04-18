from pathlib import Path

import yaml

from med_core.configs import load_config


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "project_name": "smurf-mainline",
                "experiment_name": "smurf_mvi_demo",
                "seed": 42,
                "device": "cpu",
                "data": {
                    "dataset_type": "three_phase_ct_tabular",
                    "csv_path": "data/demo/cases.csv",
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
                    "clinical_preprocessing": {
                        "normalize": True,
                        "strategy": "zero_with_mask",
                    },
                },
                "model": {
                    "model_type": "three_phase_ct_fusion",
                    "num_classes": 2,
                    "phase_feature_dim": 16,
                    "share_phase_encoder": False,
                    "phase_fusion_type": "concatenate",
                    "doctor_interest": {
                        "enabled": True,
                        "hidden_channels": 8,
                        "temperature": 6.0,
                    },
                    "topk_focus": {
                        "enabled": True,
                        "k": 3,
                        "patch_size": [4, 4, 4],
                        "projection_dim": 16,
                    },
                    "phase_encoder": {
                        "base_channels": 12,
                        "num_blocks": 3,
                        "dropout": 0.1,
                    },
                    "phase_fusion": {
                        "mode": "gated",
                        "hidden_dim": 20,
                    },
                    "use_risk_head": True,
                    "tabular": {
                        "hidden_dims": [16],
                        "output_dim": 8,
                        "dropout": 0.1,
                    },
                    "fusion": {
                        "fusion_type": "gated",
                        "hidden_dim": 12,
                        "dropout": 0.1,
                    },
                },
                "training": {
                    "num_epochs": 1,
                    "use_progressive_training": False,
                    "mixed_precision": False,
                    "doctor_interest_loss": {
                        "cam_align_weight": 0.05,
                        "consistency_weight": 0.02,
                        "sparse_weight": 0.01,
                        "diverse_weight": 0.01,
                        "body_prior_weight": 0.02,
                    },
                    "optimizer": {
                        "optimizer": "adam",
                        "learning_rate": 0.0005,
                        "weight_decay": 0.0,
                    },
                    "scheduler": {"scheduler": "none"},
                },
                "logging": {"output_dir": "outputs/smurf_mainline_test"},
                "explainability": {
                    "export_phase_importance": True,
                    "export_case_explanations": True,
                    "heatmap_ready": True,
                    "export_doctor_interest_maps": True,
                    "build_results_split": "train",
                    "min_global_importance_samples": 5,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_three_phase_mainline_config_loads_new_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "smurf_mainline.yaml"
    _write_config(config_path)

    config = load_config(config_path)

    assert config.data.dataset_type == "three_phase_ct_tabular"
    assert config.data.phase_dir_columns["arterial"] == "arterial_series_dir"
    assert config.data.clinical_feature_columns == ["age", "sex", "bmi"]
    assert config.data.target_shape == [4, 8, 8]
    assert config.model.model_type == "three_phase_ct_fusion"
    assert config.model.phase_feature_dim == 16
    assert config.model.share_phase_encoder is False
    assert config.model.phase_fusion_type == "gated"
    assert config.model.doctor_interest.enabled is True
    assert config.model.doctor_interest.hidden_channels == 8
    assert config.model.doctor_interest.temperature == 6.0
    assert config.model.topk_focus.enabled is True
    assert config.model.topk_focus.k == 3
    assert config.training.doctor_interest_loss.cam_align_weight == 0.05
    assert config.training.doctor_interest_loss.body_prior_weight == 0.02
    assert config.model.phase_encoder.base_channels == 12
    assert config.model.phase_fusion.mode == "gated"
    assert config.data.clinical_preprocessing.normalize is True
    assert config.data.clinical_preprocessing.strategy == "zero_with_mask"
    assert config.model.use_risk_head is True
    assert config.explainability.export_phase_importance is True
    assert config.explainability.heatmap_ready is True
    assert config.explainability.export_doctor_interest_maps is True
    assert config.explainability.build_results_split == "train"
    assert config.explainability.min_global_importance_samples == 5


def test_three_phase_mainline_config_roundtrips_to_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "smurf_mainline.yaml"
    _write_config(config_path)

    payload = load_config(config_path).to_dict()

    assert payload["data"]["dataset_type"] == "three_phase_ct_tabular"
    assert payload["model"]["model_type"] == "three_phase_ct_fusion"
    assert payload["model"]["phase_feature_dim"] == 16
    assert payload["model"]["phase_encoder"]["base_channels"] == 12
    assert payload["model"]["phase_fusion"]["mode"] == "gated"
    assert payload["model"]["doctor_interest"]["enabled"] is True
    assert payload["model"]["doctor_interest"]["hidden_channels"] == 8
    assert payload["model"]["topk_focus"]["enabled"] is True
    assert payload["model"]["topk_focus"]["k"] == 3
    assert payload["training"]["doctor_interest_loss"]["cam_align_weight"] == 0.05
    assert payload["explainability"]["export_doctor_interest_maps"] is True
    assert payload["data"]["clinical_preprocessing"]["strategy"] == "zero_with_mask"
    assert payload["explainability"]["export_case_explanations"] is True
    assert payload["explainability"]["build_results_split"] == "train"
    assert payload["explainability"]["min_global_importance_samples"] == 5


def test_only_one_canonical_three_phase_demo_config_exists() -> None:
    config_paths = sorted(Path("configs/demo").glob("three_phase_ct_mvi*.yaml"))

    assert config_paths == [Path("configs/demo/three_phase_ct_mvi_dr_z.yaml")]


def test_dr_z_demo_smurf_config_uses_mainline_schema() -> None:
    config = load_config("configs/demo/three_phase_ct_mvi_dr_z.yaml")

    assert config.data.dataset_type == "three_phase_ct_tabular"
    assert config.model.model_type == "three_phase_ct_fusion"
    assert config.model.phase_fusion.mode == "gated"
    assert config.model.doctor_interest.enabled is True
    assert config.model.topk_focus.enabled is True
    assert config.training.doctor_interest_loss.cam_align_weight == 0.05
    assert config.explainability.export_doctor_interest_maps is True
    assert config.data.clinical_preprocessing.strategy == "zero_with_mask"
    assert config.explainability.build_results_split == "test"
    assert config.explainability.min_global_importance_samples == 5
    assert config.logging.output_dir == "outputs/three_phase_ct_mvi_dr_z"
