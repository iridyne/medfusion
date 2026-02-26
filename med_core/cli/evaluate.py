"""Evaluation command implementation."""

import argparse
import logging
import sys
from pathlib import Path

import torch

from med_core.backbones import (
    create_tabular_backbone,
    create_vision_backbone,
)
from med_core.configs import load_config
from med_core.datasets import (
    MedicalMultimodalDataset,
    get_val_transforms,
    split_dataset,
)
from med_core.evaluation import (
    calculate_binary_metrics,
    generate_evaluation_report,
)
from med_core.fusion import (
    MultiModalFusionModel,
    create_fusion_module,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate() -> None:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a medical multimodal model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file used for training",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results", help="Output directory",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test", "train"],
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading data...")
    try:
        val_transform = get_val_transforms(image_size=config.data.image_size)
        full_dataset, _ = MedicalMultimodalDataset.from_csv(
            csv_path=config.data.csv_path,
            image_dir=config.data.image_dir,
            image_column=config.data.image_path_column,
            target_column=config.data.target_column,
            numerical_features=config.data.numerical_features,
            categorical_features=config.data.categorical_features,
            handle_missing="fill_mean",
        )
        train_ds, val_ds, test_ds = split_dataset(
            full_dataset,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            random_seed=config.data.random_seed,
        )
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Select dataset
    if args.split == "test":
        dataset = test_ds
    elif args.split == "val":
        dataset = val_ds
    else:
        dataset = train_ds

    dataset.transform = val_transform
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.data.batch_size, shuffle=False,
    )

    # Build Model structure
    logger.info("Rebuilding model architecture...")
    vision_backbone = create_vision_backbone(
        backbone_name=config.model.vision.backbone,
        feature_dim=config.model.vision.feature_dim,
        attention_type=config.model.vision.attention_type,
    )
    tabular_dim = dataset.get_tabular_dim()
    tabular_backbone = create_tabular_backbone(
        input_dim=tabular_dim,
        output_dim=config.model.tabular.output_dim,
        hidden_dims=config.model.tabular.hidden_dims,
    )
    fusion_module = create_fusion_module(
        fusion_type=config.model.fusion.fusion_type,
        vision_dim=config.model.vision.feature_dim,
        tabular_dim=config.model.tabular.output_dim,
        output_dim=config.model.fusion.hidden_dim,
    )
    model = MultiModalFusionModel(
        vision_backbone=vision_backbone,
        tabular_backbone=tabular_backbone,
        fusion_module=fusion_module,
        num_classes=config.model.num_classes,
        use_auxiliary_heads=config.model.use_auxiliary_heads,
    )

    logger.info(f"Loading weights from {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Evaluation Loop
    logger.info("Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, tabular, labels in dataloader:
            images = images.to(device)
            tabular = tabular.to(device)

            outputs = model(images, tabular)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Handle multiclass probabilities
            if config.model.num_classes == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    if config.model.num_classes == 2:
        metrics = calculate_binary_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=all_probs,
        )
    else:
        from med_core.evaluation import calculate_multiclass_metrics

        metrics = calculate_multiclass_metrics(
            y_true=all_labels,
            y_pred=all_preds,
        )

    # Generate Report
    report_path = generate_evaluation_report(
        metrics=metrics,
        output_dir=args.output_dir,
        experiment_name=config.experiment_name,
        config=config.to_dict(),
    )

    logging.info(f"Evaluation complete. Report saved to {report_path}")
