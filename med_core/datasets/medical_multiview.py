"""
Medical multi-view dataset implementation.

Extends MedicalMultimodalDataset to support multiple images per patient.
"""

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from med_core.datasets.multiview_base import BaseMultiViewDataset
from med_core.datasets.multiview_types import MultiViewConfig, ViewDict

logger = logging.getLogger(__name__)


class MedicalMultiViewDataset(BaseMultiViewDataset):
    """
    Dataset for medical multi-view multimodal learning.

    Handles loading of:
    - Multiple medical images per patient (e.g., different CT views, time series)
    - Clinical/tabular data from CSV/Excel files
    - Patient metadata and labels

    Features:
    - Supports variable number of views per patient
    - Handles missing views gracefully
    - Automatic feature preprocessing
    - Compatible with existing MedicalMultimodalDataset workflows

    Example:
        >>> config = MultiViewConfig(
        ...     view_names=["axial", "coronal", "sagittal"],
        ...     required_views=["axial"],
        ... )
        >>> dataset, scaler = MedicalMultiViewDataset.from_csv_multiview(
        ...     csv_path="data/patients.csv",
        ...     image_dir="data/images",
        ...     view_columns={"axial": "axial_path", "coronal": "coronal_path"},
        ...     target_column="diagnosis",
        ...     view_config=config,
        ... )
    """

    def __init__(
        self,
        image_paths: list[ViewDict],
        tabular_data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        view_config: MultiViewConfig,
        transform: Any = None,
        target_transform: Any = None,
        feature_names: list[str] | None = None,
        patient_ids: list[str] | None = None,
    ):
        """
        Initialize medical multi-view dataset.

        Args:
            image_paths: List of ViewDict, one per patient
            tabular_data: Preprocessed tabular features (N, num_features)
            labels: Target labels (N,)
            view_config: Multi-view configuration
            transform: Image transformations
            target_transform: Label transformations
            feature_names: Names of tabular features
            patient_ids: Optional patient identifiers
        """
        super().__init__(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            view_config=view_config,
            transform=transform,
            target_transform=target_transform,
        )

        self.feature_names = feature_names or []
        self.patient_ids = patient_ids or []

    def load_image(self, path: Path) -> Image.Image:
        """
        Load a medical image from disk.

        Args:
            path: Path to the image file

        Returns:
            PIL Image in RGB mode
        """
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path)

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    @classmethod
    def from_csv_multiview(
        cls,
        csv_path: str | Path,
        image_dir: str | Path,
        view_columns: dict[str, str],
        target_column: str = "label",
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        patient_id_column: str | None = None,
        view_config: MultiViewConfig | None = None,
        transform: Any = None,
        target_transform: Any = None,
        normalize_features: bool = True,
        handle_missing: Literal["drop", "fill_mean", "fill_zero"] = "fill_mean",
        scaler: StandardScaler | None = None,
    ) -> tuple["MedicalMultiViewDataset", StandardScaler | None]:
        """
        Create multi-view dataset from CSV file.

        This is the primary factory method for creating multi-view datasets
        from tabular data files with associated images.

        Args:
            csv_path: Path to CSV file containing patient data
            image_dir: Directory containing medical images
            view_columns: Mapping from view name to CSV column name
                Example: {"axial": "axial_path", "coronal": "coronal_path"}
            target_column: Column name containing target labels
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            patient_id_column: Optional column for patient identifiers
            view_config: Multi-view configuration (auto-created if None)
            transform: Image transformations
            target_transform: Label transformations
            normalize_features: Whether to normalize numerical features
            handle_missing: Strategy for handling missing tabular values
            scaler: Pre-fitted scaler (for val/test sets)

        Returns:
            Tuple of (dataset, fitted_scaler)

        Example:
            >>> config = MultiViewConfig(view_names=["axial", "coronal", "sagittal"])
            >>> dataset, scaler = MedicalMultiViewDataset.from_csv_multiview(
            ...     csv_path="data/ct_patients.csv",
            ...     image_dir="data/ct_images",
            ...     view_columns={
            ...         "axial": "axial_image",
            ...         "coronal": "coronal_image",
            ...         "sagittal": "sagittal_image",
            ...     },
            ...     target_column="diagnosis",
            ...     numerical_features=["age", "bmi"],
            ...     view_config=config,
            ... )
        """
        csv_path = Path(csv_path)
        image_dir = Path(image_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from {csv_path}")

        # Validate required columns
        required_cols = [target_column] + list(view_columns.values())
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")

        # Create view config if not provided
        if view_config is None:
            view_config = MultiViewConfig(
                view_names=list(view_columns.keys()),
                required_views=[],  # No required views by default
                handle_missing="zero",
            )

        # Validate view names match
        if set(view_columns.keys()) != set(view_config.view_names):
            raise ValueError(
                f"View columns {set(view_columns.keys())} don't match "
                f"config view names {set(view_config.view_names)}"
            )

        # Handle missing tabular values
        df = cls._handle_missing_values(
            df,
            numerical_features or [],
            categorical_features or [],
            strategy=handle_missing,
        )

        # Build multi-view image paths
        image_paths_list: list[ViewDict] = []
        valid_indices = []

        for idx, row in df.iterrows():
            view_dict: ViewDict = {}
            has_any_view = False

            for view_name, col_name in view_columns.items():
                img_name = row[col_name]

                # Handle missing view (NaN or empty string)
                if pd.isna(img_name) or str(img_name).strip() == "":
                    view_dict[view_name] = None
                    continue

                # Build image path
                if Path(img_name).is_absolute():
                    img_path = Path(img_name)
                else:
                    img_path = image_dir / img_name

                if img_path.exists():
                    view_dict[view_name] = img_path
                    has_any_view = True
                else:
                    logger.warning(f"Image not found: {img_path}")
                    view_dict[view_name] = None

            # Only include samples with at least one valid view
            if has_any_view:
                image_paths_list.append(view_dict)
                valid_indices.append(idx)

        # Filter dataframe to valid indices
        df = df.iloc[valid_indices].reset_index(drop=True)
        logger.info(f"Found {len(image_paths_list)} valid multi-view samples")

        # Log view availability statistics
        view_counts = dict.fromkeys(view_columns.keys(), 0)
        for view_dict in image_paths_list:
            for view_name, path in view_dict.items():
                if path is not None:
                    view_counts[view_name] += 1

        logger.info("View availability:")
        for view_name, count in view_counts.items():
            percentage = 100 * count / len(image_paths_list)
            logger.info(f"  {view_name}: {count}/{len(image_paths_list)} ({percentage:.1f}%)")

        # Extract and preprocess tabular features
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
            image_paths=image_paths_list,
            tabular_data=tabular_data,
            labels=labels,
            view_config=view_config,
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
            feature_cols = numerical_features + categorical_features
            df = df.dropna(subset=feature_cols)

        elif strategy == "fill_mean":
            for col in numerical_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

            for col in categorical_features:
                if col in df.columns:
                    mode = df[col].mode()
                    fill_value = mode.iloc[0] if len(mode) > 0 else 0
                    df[col] = df[col].fillna(fill_value)

        elif strategy == "fill_zero":
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
        """Prepare tabular features from dataframe."""
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

        # Process categorical features
        for cat_col in categorical_features:
            if cat_col in df.columns:
                cat_data = pd.Categorical(df[cat_col]).codes
                feature_arrays.append(cat_data.reshape(-1, 1).astype(np.float32))
                feature_names.append(cat_col)

        # Combine all features
        if feature_arrays:
            tabular_data = np.hstack(feature_arrays)
        else:
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
