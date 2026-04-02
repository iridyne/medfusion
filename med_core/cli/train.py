"""Training command implementation."""

import argparse
import json
import logging
import shutil
import sys
from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from med_core.backbones import (
    create_tabular_backbone,
    create_vision_backbone,
)
from med_core.configs import load_config
from med_core.datasets import (
    MedicalMultimodalDataset,
    ThreePhaseCTCaseDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    split_dataset,
)
from med_core.fusion import (
    MultiModalFusionModel,
    create_fusion_module,
)
from med_core.models import ThreePhaseCTFusionModel
from med_core.output_layout import RunOutputLayout
from med_core.shared.preprocessing.clinical import ClinicalFeaturePreprocessor
from med_core.shared.model_utils import ModelCheckpoint
from med_core.trainers import MultimodalTrainer
from med_core.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _read_manifest_dataframe(csv_path: str, patient_id_column: str | None) -> pd.DataFrame:
    read_csv_kwargs = {}
    if patient_id_column:
        read_csv_kwargs["dtype"] = {patient_id_column: "string"}
    return pd.read_csv(csv_path, **read_csv_kwargs)


def _extract_clinical_rows(
    dataframe: pd.DataFrame,
    clinical_feature_columns: Sequence[str],
) -> list[list[float | None]]:
    return [
        [
            None if pd.isna(row[column]) else float(row[column])
            for column in clinical_feature_columns
        ]
        for row in dataframe.to_dict(orient="records")
    ]


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_optimizer(
    model: MultiModalFusionModel,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
):
    """Create optimizer from config."""
    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(
    optimizer,
    scheduler_name: str,
    num_epochs: int,
    train_loader_length: int,
    min_lr: float,
    step_size: int,
    gamma: float,
    patience: int,
    factor: float,
    mode: str,
):
    """Create learning rate scheduler from config."""
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    if scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
        )
    if scheduler_name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            epochs=max(num_epochs, 1),
            steps_per_epoch=max(train_loader_length, 1),
        )
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs, 1),
        eta_min=min_lr,
    )


def _split_indices(
    dataset_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()

    train_count = int(round(dataset_size * train_ratio))
    val_count = int(round(dataset_size * val_ratio))
    if train_count <= 0 and dataset_size > 0:
        train_count = 1
    if train_count + val_count > dataset_size:
        val_count = max(dataset_size - train_count, 0)
    test_count = max(dataset_size - train_count - val_count, 0)

    train_indices = permutation[:train_count]
    val_indices = permutation[train_count : train_count + val_count]
    test_indices = permutation[
        train_count + val_count : train_count + val_count + test_count
    ]
    return train_indices, val_indices, test_indices


def _train_three_phase_ct(config, run_layout: RunOutputLayout) -> None:
    set_seed(int(config.seed), deterministic=bool(config.deterministic))
    device = _resolve_device(config.device)

    dataframe = _read_manifest_dataframe(
        config.data.resolved_csv_path,
        config.data.patient_id_column,
    )
    train_indices, val_indices, _ = _split_indices(
        dataset_size=len(dataframe),
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.data.random_seed,
    )
    train_dataframe = dataframe.iloc[train_indices].reset_index(drop=True)
    val_dataframe = dataframe.iloc[
        val_indices if val_indices else train_indices
    ].reset_index(drop=True)

    preprocessor = ClinicalFeaturePreprocessor(
        strategy=config.data.clinical_preprocessing.strategy,
        normalize=config.data.clinical_preprocessing.normalize,
    )
    if preprocessor.normalize:
        preprocessor.fit(
            _extract_clinical_rows(
                train_dataframe,
                config.data.clinical_feature_columns,
            )
        )
    clinical_preprocessing_path = (
        run_layout.artifacts_dir / "clinical_preprocessing.json"
    )
    clinical_preprocessing_path.write_text(
        json.dumps(preprocessor.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_subset = ThreePhaseCTCaseDataset.from_manifest_dataframe(
        train_dataframe,
        phase_dir_columns=config.data.phase_dir_columns,
        clinical_feature_columns=config.data.clinical_feature_columns,
        target_column=config.data.target_column,
        patient_id_column=config.data.patient_id_column or "case_id",
        target_shape=tuple(config.data.target_shape or [16, 64, 64]),
        window_preset=config.data.window_preset,
        clinical_preprocessor=preprocessor,
    )
    val_subset = ThreePhaseCTCaseDataset.from_manifest_dataframe(
        val_dataframe,
        phase_dir_columns=config.data.phase_dir_columns,
        clinical_feature_columns=config.data.clinical_feature_columns,
        target_column=config.data.target_column,
        patient_id_column=config.data.patient_id_column or "case_id",
        target_shape=tuple(config.data.target_shape or [16, 64, 64]),
        window_preset=config.data.window_preset,
        clinical_preprocessor=preprocessor,
    )

    clinical_mask_dim = (
        len(config.data.clinical_feature_columns)
        if config.data.clinical_preprocessing.strategy == "zero_with_mask"
        else 0
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda" and config.data.pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=max(config.data.batch_size, 1),
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda" and config.data.pin_memory,
    )

    model = ThreePhaseCTFusionModel(
        phase_feature_dim=config.model.phase_feature_dim,
        clinical_input_dim=len(config.data.clinical_feature_columns),
        clinical_mask_dim=clinical_mask_dim,
        clinical_hidden_dim=config.model.tabular.output_dim,
        fusion_hidden_dim=config.model.fusion.hidden_dim,
        num_classes=config.model.num_classes,
        phase_fusion_type=config.model.phase_fusion.mode,
        share_phase_encoder=config.model.share_phase_encoder,
        freeze_phase_encoder=config.model.vision.freeze_backbone,
        use_risk_head=config.model.use_risk_head,
        phase_encoder_base_channels=config.model.phase_encoder.base_channels,
        phase_encoder_num_blocks=config.model.phase_encoder.num_blocks,
        phase_encoder_dropout=config.model.phase_encoder.dropout,
        phase_encoder_norm_type=config.model.phase_encoder.norm,
    ).to(device)
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=config.training.optimizer.optimizer,
        learning_rate=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        momentum=config.training.optimizer.momentum,
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_manager = ModelCheckpoint(
        save_dir=run_layout.checkpoints_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    history_entries: list[dict[str, float | int | bool]] = []
    best_val_loss = float("inf")

    for epoch in range(max(config.training.num_epochs, 1)):
        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                arterial=batch["arterial"].to(device),
                portal=batch["portal"].to(device),
                noncontrast=batch["noncontrast"].to(device),
                clinical=batch["clinical"].to(device),
                clinical_missing_mask=(
                    batch["clinical_missing_mask"].to(device)
                    if clinical_mask_dim > 0
                    else None
                ),
            )
            loss = criterion(outputs["logits"], batch["label"].to(device))
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())

        model.eval()
        val_loss_sum = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                outputs = model(
                    arterial=batch["arterial"].to(device),
                    portal=batch["portal"].to(device),
                    noncontrast=batch["noncontrast"].to(device),
                    clinical=batch["clinical"].to(device),
                    clinical_missing_mask=(
                        batch["clinical_missing_mask"].to(device)
                        if clinical_mask_dim > 0
                        else None
                    ),
                )
                loss = criterion(outputs["logits"], batch["label"].to(device))
                val_loss_sum += float(loss.item())

        train_loss = train_loss_sum / max(len(train_loader), 1)
        val_loss = val_loss_sum / max(len(val_loader), 1)
        checkpoint_manager.save(
            model=model,
            epoch=epoch + 1,
            metrics={"val_loss": val_loss},
            optimizer=optimizer,
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            history_entry["best_so_far"] = True
        history_entries.append(history_entry)

    run_layout.history_path.write_text(
        json.dumps({"entries": history_entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_checkpoint_path = run_layout.checkpoints_dir / "best.pth"
    if checkpoint_manager.best_model_path is not None:
        shutil.copy2(checkpoint_manager.best_model_path, best_checkpoint_path)
    else:
        shutil.copy2(run_layout.checkpoints_dir / "last.pth", best_checkpoint_path)


def train(
    argv: Sequence[str] | None = None,
    prog: str = "med-train",
) -> None:
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train a medical multimodal model",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    run_layout = RunOutputLayout(config.logging.output_dir).ensure_exists()

    logger.info("Starting experiment: %s", config.experiment_name)
    logger.info("Output directory: %s", run_layout.root_dir)

    if (
        config.data.dataset_type == "three_phase_ct_tabular"
        or config.model.model_type == "three_phase_ct_fusion"
    ):
        logger.info("Using native three-phase CT + tabular mainline training path")
        _train_three_phase_ct(config, run_layout)
        logger.info("Training completed successfully!")
        return

    logger.info("Loading data...")
    train_transform = get_train_transforms(
        image_size=config.data.image_size,
        augmentation_strength=config.data.augmentation_strength,
    )
    val_transform = get_val_transforms(image_size=config.data.image_size)

    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.resolved_csv_path,
        image_dir=config.data.resolved_image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        patient_id_column=config.data.patient_id_column,
        transform=None,
    )

    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )

    train_ds.transform = train_transform
    val_ds.transform = val_transform
    test_ds.transform = val_transform

    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    logger.info("Building model...")
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=config.model.vision.pretrained,
        freeze=config.model.vision.freeze_backbone,
        feature_dim=config.model.vision.feature_dim,
        dropout=config.model.vision.dropout,
        attention_type=config.model.vision.attention_type,
        enable_attention_supervision=config.model.vision.enable_attention_supervision,
    )

    tabular_dim = train_ds.get_tabular_dim()
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
        dropout=config.model.tabular.dropout,
    )

    fusion_kwargs = {
        "dropout": config.model.fusion.dropout,
    }
    if config.model.fusion.fusion_type in {"attention", "cross_attention"}:
        fusion_kwargs["num_heads"] = config.model.fusion.num_heads

    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
        **fusion_kwargs,
    )

    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )

    optimizer = _build_optimizer(
        model=model,
        optimizer_name=config.training.optimizer.optimizer,
        learning_rate=config.training.optimizer.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        momentum=config.training.optimizer.momentum,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=config.training.scheduler.scheduler,
        num_epochs=config.training.num_epochs,
        train_loader_length=len(dataloaders["train"]),
        min_lr=config.training.scheduler.min_lr,
        step_size=config.training.scheduler.step_size,
        gamma=config.training.scheduler.gamma,
        patience=config.training.scheduler.patience,
        factor=config.training.scheduler.factor,
        mode=config.training.mode,
    )
    logger.info(
        "Using optimizer=%s, scheduler=%s",
        config.training.optimizer.optimizer,
        config.training.scheduler.scheduler,
    )

    trainer = MultimodalTrainer(
        config=config,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
    )

    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training completed successfully!")
