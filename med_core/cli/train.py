"""Training command implementation."""

import argparse
import logging
import sys
from collections.abc import Sequence

from torch import optim

from med_core.backbones import (
    create_tabular_backbone,
    create_vision_backbone,
)
from med_core.configs import load_config
from med_core.datasets import (
    MedicalMultimodalDataset,
    create_dataloaders,
    get_train_transforms,
    get_val_transforms,
    split_dataset,
)
from med_core.fusion import (
    MultiModalFusionModel,
    create_fusion_module,
)
from med_core.trainers import MultimodalTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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
        "--config", type=str, required=True, help="Path to YAML configuration file",
    )
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    args = parser.parse_args(argv)

    # 1. Load Configuration
    config = load_config(args.config)
    if args.output_dir:
        config.logging.output_dir = args.output_dir

    logger.info(f"Starting experiment: {config.experiment_name}")
    logger.info(f"Output directory: {config.logging.output_dir}")

    # 2. Setup Data
    logger.info("Loading data...")
    # Transformations
    train_transform = get_train_transforms(
        image_size=config.data.image_size,
        augmentation_strength=config.data.augmentation_strength,
    )
    val_transform = get_val_transforms(image_size=config.data.image_size)

    # Load full dataset
    full_dataset, _ = MedicalMultimodalDataset.from_csv(
        csv_path=config.data.csv_path,
        image_dir=config.data.image_dir,
        image_column=config.data.image_path_column,
        target_column=config.data.target_column,
        numerical_features=config.data.numerical_features,
        categorical_features=config.data.categorical_features,
        patient_id_column=config.data.patient_id_column,
        transform=None,  # Transforms applied later or handled in split
    )

    # Split dataset
    train_ds, val_ds, test_ds = split_dataset(
        full_dataset,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=config.data.random_seed,
    )

    # Apply transforms to subsets
    train_ds.transform = train_transform
    val_ds.transform = val_transform
    test_ds.transform = val_transform

    # Create dataloaders
    dataloaders = create_dataloaders(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # 3. Setup Model
    logger.info("Building model...")
    # Vision Backbone
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        pretrained=config.model.vision.pretrained,
        freeze=config.model.vision.freeze_backbone,
        feature_dim=config.model.vision.feature_dim,
        dropout=config.model.vision.dropout,
        attention_type=config.model.vision.attention_type,
        enable_attention_supervision=config.model.vision.enable_attention_supervision,
    )

    # Tabular Backbone
    tabular_dim = train_ds.get_tabular_dim()
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
        dropout=config.model.tabular.dropout,
    )

    # Fusion Module
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

    # Complete Model
    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )

    # 4. Optimizer & Scheduler
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

    # 5. Trainer
    trainer = MultimodalTrainer(
        config=config,
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # 6. Train
    logger.info("Starting training loop...")
    trainer.train()
    logger.info("Training completed successfully!")
