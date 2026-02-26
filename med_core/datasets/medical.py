"""
Medical multimodal dataset implementation.

Provides a concrete implementation for loading medical images
along with tabular (structured) clinical data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from med_core.datasets.base import BaseMultimodalDataset
from med_core.datasets.data_cleaner import DataCleaner

logger = logging.getLogger(__name__)


class MedicalMultimodalDataset(BaseMultimodalDataset):
    """
    Dataset for medical multimodal learning.

    Handles loading of:
    - Medical images (JPEG, PNG, TIFF, etc.)
    - Clinical/tabular data from CSV/Excel files
    - Patient metadata and labels

    Features:
    - Automatic feature preprocessing (normalization, encoding)
    - Support for various image formats
    - Handles missing data gracefully
    - Caches processed data for faster loading

    Example:
        >>> dataset = MedicalMultimodalDataset.from_csv(
        ...     csv_path="data/patients.csv",
        ...     image_dir="data/images",
        ...     image_column="image_path",
        ...     target_column="diagnosis",
        ...     numerical_features=["age", "blood_pressure"],
        ...     categorical_features=["gender", "smoking"],
        ... )
    """

    def __init__(
        self,
        image_paths: list[str | Path],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        transform: Any = None,
        target_transform: Any = None,
        feature_names: list[str] | None = None,
        patient_ids: list[str] | None = None,
    ):
        """
        Initialize medical multimodal dataset.

        Args:
            image_paths: List of paths to medical images
            tabular_data: Preprocessed tabular features (N, num_features)
            labels: Target labels (N,)
            transform: Image transformations
            target_transform: Label transformations
            feature_names: Names of tabular features (for interpretability)
            patient_ids: Optional patient identifiers
        """
        super().__init__(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
        )

        self.feature_names = feature_names or []
        self.patient_ids = patient_ids or []

    def load_image(self, path: Path) -> Image.Image:
        """
        Load a medical image from disk.

        Handles common medical image formats.
        For DICOM support, use MedicalMultimodalDataset.with_dicom_support().

        Args:
            path: Path to the image file

        Returns:
            PIL Image in RGB mode
        """
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image and convert to RGB
        image = Image.open(path)

        # Convert grayscale or RGBA to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        image_dir: str | Path,
        image_column: str = "image_path",
        target_column: str = "label",
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        patient_id_column: str | None = None,
        transform: Any = None,
        target_transform: Any = None,
        normalize_features: bool = True,
        handle_missing: Literal["drop", "fill_mean", "fill_zero"] = "fill_mean",
        scaler: StandardScaler | None = None,
        data_cleaner: DataCleaner | None = None,
    ) -> tuple["MedicalMultimodalDataset", StandardScaler | None]:
        """
        Create dataset from CSV file.

        This is the primary factory method for creating datasets
        from tabular data files with associated images.

        Args:
            csv_path: Path to CSV file containing patient data
            image_dir: Directory containing medical images
            image_column: Column name containing image file names/paths
            target_column: Column name containing target labels
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            patient_id_column: Optional column for patient identifiers
            transform: Image transformations
            target_transform: Label transformations
            normalize_features: Whether to normalize numerical features
            handle_missing: Strategy for handling missing values
            scaler: Pre-fitted scaler (for val/test sets)
            data_cleaner: Optional DataCleaner instance for custom cleaning logic

        Returns:
            Tuple of (dataset, fitted_scaler)
        """
        csv_path = Path(csv_path)
        image_dir = Path(image_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from {csv_path}")

        # Validate required columns
        required_cols = [image_column, target_column]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        # Use DataCleaner if provided, otherwise use default cleaning logic
        if data_cleaner is not None:
            # Clean data using DataCleaner
            df = data_cleaner.handle_missing_values(df)
        else:
            # Fallback to legacy methods for backward compatibility
            df = cls._handle_missing_values(
                df,
                numerical_features or [],
                categorical_features or [],
                strategy=handle_missing,
            )

        # Build image paths
        image_paths = []
        valid_indices = []

        for idx, row in df.iterrows():
            img_name = row[image_column]
            # Handle both full paths and just filenames
            if Path(img_name).is_absolute():
                img_path = Path(img_name)
            else:
                img_path = image_dir / img_name

            # Resolve to absolute path and validate it's within image_dir
            try:
                img_path = img_path.resolve()
                # Ensure the resolved path is within image_dir (prevent path traversal)
                if not Path(img_name).is_absolute():
                    image_dir_resolved = image_dir.resolve()
                    if not str(img_path).startswith(str(image_dir_resolved)):
                        logger.warning(f"Path traversal attempt detected: {img_name}")
                        continue
            except (OSError, RuntimeError) as e:
                logger.warning(f"Invalid path {img_name}: {e}")
                continue

            if img_path.exists():
                image_paths.append(img_path)
                valid_indices.append(idx)
            else:
                logger.warning(f"Image not found: {img_path}")

        # Filter dataframe to valid indices
        df = df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Found {len(image_paths)} valid image-data pairs")

        # Extract and preprocess tabular features
        if data_cleaner is not None:
            tabular_data, feature_names, scaler = data_cleaner.prepare_tabular_features(
                df, scaler,
            )
        else:
            tabular_data, feature_names, scaler = cls._prepare_tabular_features(
                df=df,
                numerical_features=numerical_features or [],
                categorical_features=categorical_features or [],
                normalize=normalize_features,
                scaler=scaler,
            )

        # Extract labels
        labels = df[target_column].values

        # Extract patient IDs if available
        patient_ids = None
        if patient_id_column and patient_id_column in df.columns:
            patient_ids = df[patient_id_column].astype(str).tolist()

        dataset = cls(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            transform=transform,
            target_transform=target_transform,
            feature_names=feature_names,
            patient_ids=patient_ids,
        )

        return dataset, scaler

    @staticmethod
    def _handle_missing_values(
        df: pd.DataFrame,
        numerical_features: list[str],
        categorical_features: list[str],
        strategy: str = "fill_mean",
    ) -> pd.DataFrame:
        """Handle missing values in the dataframe."""
        df = df.copy()

        if strategy == "drop":
            # Drop rows with any missing values in feature columns
            feature_cols = numerical_features + categorical_features
            df = df.dropna(subset=feature_cols)

        elif strategy == "fill_mean":
            # Fill numerical with mean, categorical with mode
            for col in numerical_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

            for col in categorical_features:
                if col in df.columns:
                    mode = df[col].mode()
                    fill_value = mode.iloc[0] if len(mode) > 0 else 0
                    df[col] = df[col].fillna(fill_value)

        elif strategy == "fill_zero":
            # Fill all missing with zero
            feature_cols = numerical_features + categorical_features
            for col in feature_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        return df

    @staticmethod
    def _prepare_tabular_features(
        df: pd.DataFrame,
        numerical_features: list[str],
        categorical_features: list[str],
        normalize: bool = True,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, list[str], StandardScaler | None]:
        """
        Prepare tabular features from dataframe.

        Handles numerical normalization and categorical encoding.
        """
        feature_arrays = []
        feature_names = []

        # Process numerical features
        if numerical_features:
            numerical_data = df[numerical_features].values.astype(np.float32)

            if normalize:
                if scaler is None:
                    scaler = StandardScaler()
                    numerical_data = scaler.fit_transform(numerical_data)
                else:
                    numerical_data = scaler.transform(numerical_data)

            feature_arrays.append(numerical_data)
            feature_names.extend(numerical_features)

        # Process categorical features (simple label encoding for now)
        for cat_col in categorical_features:
            if cat_col in df.columns:
                # Convert to numeric codes
                cat_data = pd.Categorical(df[cat_col]).codes
                feature_arrays.append(cat_data.reshape(-1, 1).astype(np.float32))
                feature_names.append(cat_col)

        # Combine all features
        if feature_arrays:
            tabular_data = np.hstack(feature_arrays)
        else:
            # No features specified - create dummy feature
            tabular_data = np.zeros((len(df), 1), dtype=np.float32)
            feature_names = ["dummy"]

        return tabular_data, feature_names, scaler

    def get_feature_names(self) -> list[str]:
        """Return names of tabular features."""
        return self.feature_names

    def get_patient_id(self, idx: int) -> str | None:
        """Get patient ID for a given index."""
        if self.patient_ids and idx < len(self.patient_ids):
            return self.patient_ids[idx]
        return None


def split_dataset(
    dataset: MedicalMultimodalDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True,
) -> tuple[
    MedicalMultimodalDataset, MedicalMultimodalDataset, MedicalMultimodalDataset,
]:
    """
    Split dataset into train/val/test sets.

    Ensures stratified splitting for classification tasks.

    Args:
        dataset: Dataset to split
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        stratify: Whether to use stratified splitting

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) >= 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    n_samples = len(dataset)
    indices = list(range(n_samples))
    labels = dataset.labels.numpy() if stratify else None

    # First split: train vs (val + test)
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_seed,
        stratify=labels if stratify else None,
    )

    # Second split: val vs test
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    temp_labels = labels[temp_indices] if stratify else None

    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_relative_ratio,
        random_state=random_seed,
        stratify=temp_labels if stratify else None,
    )

    # Create subset datasets
    train_dataset = dataset.subset(train_indices)
    val_dataset = dataset.subset(val_indices)
    test_dataset = dataset.subset(test_indices)

    logger.info(
        f"Split dataset: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}",
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: MedicalMultimodalDataset,
    val_dataset: MedicalMultimodalDataset,
    test_dataset: MedicalMultimodalDataset | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampling: bool = True,
) -> dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        use_weighted_sampling: Whether to use weighted sampling for imbalanced data

    Returns:
        Dictionary of DataLoaders with keys 'train', 'val', 'test'
    """
    dataloaders = {}

    # Training dataloader with optional weighted sampling
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    else:
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    # Validation dataloader
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Test dataloader (if provided)
    if test_dataset is not None:
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return dataloaders


def create_kfold_splits(
    dataset: MedicalMultimodalDataset,
    n_folds: int = 5,
    random_seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """
    Create K-fold cross-validation splits.

    Args:
        dataset: Dataset to split
        n_folds: Number of folds
        random_seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    labels = dataset.labels.numpy()
    indices = np.arange(len(dataset))

    splits = []
    for train_idx, val_idx in kfold.split(indices, labels):
        splits.append((train_idx.tolist(), val_idx.tolist()))

    return splits


def save_split_info(
    train_dataset: MedicalMultimodalDataset,
    val_dataset: MedicalMultimodalDataset,
    test_dataset: MedicalMultimodalDataset | None,
    output_path: str | Path,
) -> None:
    """
    Save dataset split information for reproducibility.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset
        output_path: Path to save split information
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_info = {
        "train": {
            "num_samples": len(train_dataset),
            "class_distribution": train_dataset.get_class_distribution(),
            "image_paths": [str(p) for p in train_dataset.image_paths],
        },
        "val": {
            "num_samples": len(val_dataset),
            "class_distribution": val_dataset.get_class_distribution(),
            "image_paths": [str(p) for p in val_dataset.image_paths],
        },
    }

    if test_dataset is not None:
        split_info["test"] = {
            "num_samples": len(test_dataset),
            "class_distribution": test_dataset.get_class_distribution(),
            "image_paths": [str(p) for p in test_dataset.image_paths],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Split information saved to {output_path}")
