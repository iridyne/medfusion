"""Tests for configuration validation."""

from med_core.configs import (
    DataConfig,
    ExperimentConfig,
    ExplainabilityConfig,
    FusionConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TabularConfig,
    TrainingConfig,
    VisionConfig,
    validate_config,
)


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test that a valid config passes validation."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="resnet18",
                    feature_dim=128,
                    dropout=0.3,
                    attention_type="none",
                ),
                tabular=TabularConfig(
                    hidden_dims=[64],
                    output_dim=32,
                ),
                fusion=FusionConfig(
                    fusion_type="concatenate",
                    hidden_dim=96,
                ),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    learning_rate=1e-3,
                ),
                scheduler=SchedulerConfig(
                    scheduler="cosine",
                ),
            ),
            logging=LoggingConfig(
                output_dir="outputs/",
                use_tensorboard=True,
                use_wandb=False,
            ),
        )

        errors = validate_config(config)
        assert len(errors) == 0

    def test_invalid_backbone(self):
        """Test validation catches invalid backbone."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="invalid_backbone",
                    feature_dim=128,
                ),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("backbone" in e.path for e in errors)
        assert any(e.error_code == "E002" for e in errors)

    def test_runtime_supported_backbone_and_fusion_pass_validation(self):
        """Runtime-supported canonical values should also pass config validation."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="mobilenetv3_small",
                    feature_dim=128,
                    attention_type="none",
                ),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(
                    fusion_type="cross_attention",
                    hidden_dim=96,
                    num_heads=4,
                ),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(
                    optimizer="adam",
                    learning_rate=1e-3,
                ),
                scheduler=SchedulerConfig(
                    scheduler="cosine",
                ),
            ),
            logging=LoggingConfig(
                output_dir="outputs/",
                use_tensorboard=True,
                use_wandb=False,
            ),
        )

        errors = validate_config(config)
        assert len(errors) == 0

    def test_invalid_num_classes(self):
        """Test validation catches invalid num_classes."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=1,  # Invalid: must be >= 2
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("num_classes" in e.path for e in errors)
        assert any(e.error_code == "E001" for e in errors)

    def test_invalid_split_ratios(self):
        """Test validation catches invalid split ratios."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1, invalid
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("ratio" in e.path for e in errors)
        assert any(e.error_code == "E011" for e in errors)

    def test_invalid_explainability_build_results_split(self):
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                model_type="three_phase_ct_fusion",
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                patient_id_column="case_id",
                target_column="label",
                phase_dir_columns={
                    "arterial": "a",
                    "portal": "p",
                    "noncontrast": "n",
                },
                clinical_feature_columns=["age"],
                target_shape=[4, 8, 8],
                batch_size=1,
            ),
            training=TrainingConfig(
                num_epochs=1,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
            explainability=ExplainabilityConfig(build_results_split="demo"),  # type: ignore[arg-type]
        )

        errors = validate_config(config)
        assert any(e.path == "explainability.build_results_split" for e in errors)

    def test_invalid_min_global_importance_samples(self):
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                model_type="three_phase_ct_fusion",
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                patient_id_column="case_id",
                target_column="label",
                phase_dir_columns={
                    "arterial": "a",
                    "portal": "p",
                    "noncontrast": "n",
                },
                clinical_feature_columns=["age"],
                target_shape=[4, 8, 8],
                batch_size=1,
            ),
            training=TrainingConfig(
                num_epochs=1,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
            explainability=ExplainabilityConfig(min_global_importance_samples=0),
        )

        errors = validate_config(config)
        assert any(
            e.path == "explainability.min_global_importance_samples" for e in errors
        )

    def test_invalid_fusion_type(self):
        """Test validation catches invalid fusion type."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(
                    fusion_type="invalid_fusion",
                    hidden_dim=96,
                ),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("fusion_type" in e.path for e in errors)
        assert any(e.error_code == "E009" for e in errors)

    def test_three_phase_dataset_requires_phase_columns(self):
        """Three-phase config should require named phase manifest columns."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="three_phase_missing_columns",
            model=ModelConfig(
                model_type="three_phase_ct_fusion",
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=1,
                num_workers=0,
                patient_id_column="case_id",
                clinical_feature_columns=["age"],
                target_shape=[16, 64, 64],
                window_preset="liver",
            ),
            training=TrainingConfig(
                num_epochs=1,
                use_progressive_training=False,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert any(error.error_code == "E031" for error in errors)

    def test_three_phase_dataset_and_model_pairing_must_match(self):
        """Three-phase dataset should be rejected when paired with the old model type."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="three_phase_pairing_mismatch",
            model=ModelConfig(
                model_type="multimodal_fusion",
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=1,
                num_workers=0,
                patient_id_column="case_id",
                phase_dir_columns={
                    "arterial": "arterial_series_dir",
                    "portal": "portal_series_dir",
                    "noncontrast": "noncontrast_series_dir",
                },
                clinical_feature_columns=["age"],
                target_shape=[16, 64, 64],
                window_preset="liver",
            ),
            training=TrainingConfig(
                num_epochs=1,
                use_progressive_training=False,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert any(error.error_code == "E036" for error in errors)

    def test_three_phase_invalid_phase_fusion_mode(self):
        """Three-phase config should reject unsupported phase fusion modes."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="three_phase_invalid_phase_fusion",
            model=ModelConfig(
                model_type="three_phase_ct_fusion",
                num_classes=2,
                phase_fusion={
                    "mode": "invalid_mode",
                    "hidden_dim": 32,
                },
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="gated", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=1,
                num_workers=0,
                patient_id_column="case_id",
                phase_dir_columns={
                    "arterial": "arterial_series_dir",
                    "portal": "portal_series_dir",
                    "noncontrast": "noncontrast_series_dir",
                },
                clinical_feature_columns=["age"],
                target_shape=[16, 64, 64],
                window_preset="liver",
            ),
            training=TrainingConfig(
                num_epochs=1,
                use_progressive_training=False,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert any(error.path == "model.phase_fusion.mode" for error in errors)

    def test_three_phase_invalid_clinical_preprocessing_strategy(self):
        """Three-phase config should reject unsupported clinical preprocessing strategy."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="three_phase_invalid_clinical_preprocessing",
            model=ModelConfig(
                model_type="three_phase_ct_fusion",
                num_classes=2,
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="gated", hidden_dim=96),
            ),
            data=DataConfig(
                dataset_type="three_phase_ct_tabular",
                csv_path="data.csv",
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                batch_size=1,
                num_workers=0,
                patient_id_column="case_id",
                phase_dir_columns={
                    "arterial": "arterial_series_dir",
                    "portal": "portal_series_dir",
                    "noncontrast": "noncontrast_series_dir",
                },
                clinical_feature_columns=["age"],
                clinical_preprocessing={
                    "normalize": True,
                    "strategy": "unsupported",
                },
                target_shape=[16, 64, 64],
                window_preset="liver",
            ),
            training=TrainingConfig(
                num_epochs=1,
                use_progressive_training=False,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert any(
            error.path == "data.clinical_preprocessing.strategy" for error in errors
        )

    def test_progressive_training_epochs_mismatch(self):
        """Test validation catches progressive training epoch mismatch."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                use_progressive_training=True,
                stage1_epochs=10,
                stage2_epochs=10,
                stage3_epochs=10,  # Sum = 30, but num_epochs = 50
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("stage" in e.path for e in errors)
        assert any(e.error_code == "E020" for e in errors)

    def test_attention_supervision_requires_cbam(self):
        """Test validation catches attention supervision without CBAM."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="resnet18",
                    feature_dim=128,
                    attention_type="se",  # Not CBAM
                    enable_attention_supervision=False,
                ),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                use_attention_supervision=True,  # Enabled
                attention_supervision_method="mask_guided",
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any(e.error_code == "E028" for e in errors)

    def test_wandb_requires_project(self):
        """Test validation catches wandb enabled without project name."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(
                output_dir="outputs/",
                use_wandb=True,
                wandb_project=None,  # Missing project name
            ),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("wandb_project" in e.path for e in errors)
        assert any(e.error_code == "E027" for e in errors)

    def test_validation_error_has_suggestion(self):
        """Test that validation errors include helpful suggestions."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="invalid_backbone",
                    feature_dim=128,
                ),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        backbone_errors = [e for e in errors if "backbone" in e.path]
        assert len(backbone_errors) > 0
        assert backbone_errors[0].suggestion is not None
        assert "Choose from" in backbone_errors[0].suggestion

    def test_negative_batch_size(self):
        """Test validation catches negative batch size."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(backbone="resnet18", feature_dim=128),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=-1,  # Invalid
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("batch_size" in e.path for e in errors)
        assert any(e.error_code == "E012" for e in errors)

    def test_invalid_dropout_range(self):
        """Test validation catches dropout outside [0, 1)."""
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            model=ModelConfig(
                num_classes=2,
                vision=VisionConfig(
                    backbone="resnet18",
                    feature_dim=128,
                    dropout=1.5,  # Invalid: > 1
                ),
                tabular=TabularConfig(hidden_dims=[64], output_dim=32),
                fusion=FusionConfig(fusion_type="concatenate", hidden_dim=96),
            ),
            data=DataConfig(
                data_root="data/",
                csv_path="data.csv",
                image_dir="images/",
                batch_size=32,
                image_size=224,
            ),
            training=TrainingConfig(
                num_epochs=50,
                optimizer=OptimizerConfig(optimizer="adam", learning_rate=1e-3),
                scheduler=SchedulerConfig(scheduler="none"),
            ),
            logging=LoggingConfig(output_dir="outputs/"),
        )

        errors = validate_config(config)
        assert len(errors) > 0
        assert any("dropout" in e.path for e in errors)
        assert any(e.error_code == "E005" for e in errors)
