"""
Tabular Data Preprocessing Utilities
====================================
Common preprocessing functions for tabular medical data.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class TabularPreprocessor:
    """
    Preprocessing pipeline for tabular data.

    Example:
        >>> preprocessor = TabularPreprocessor()
        >>> X_train, y_train = preprocessor.fit_transform(df_train, target_col="label")
        >>> X_test, y_test = preprocessor.transform(df_test, target_col="label")
    """

    def __init__(
        self,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        scaling_method: Literal["standard", "minmax", "none"] = "standard",
    ):
        """
        Args:
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            scaling_method: Method for scaling numerical features
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.scaling_method = scaling_method

        self.label_encoders: dict[str, Any] = {}
        self.scaler = (
            StandardScaler() if scaling_method == "standard" else MinMaxScaler()
        )
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Fit preprocessor and transform data.

        Args:
            df: Input dataframe
            target_col: Target column name (optional)

        Returns:
            Tuple of (X, y) where y is None if target_col not provided
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Encode categorical features
        X_cat = self._encode_categorical(df[self.categorical_features], fit=True)

        # Scale numerical features
        X_num = self._scale_numerical(df[self.numerical_features], fit=True)

        # Combine features
        X = (
            np.hstack([X_cat, X_num])
            if X_cat.size > 0 and X_num.size > 0
            else (X_cat if X_cat.size > 0 else X_num)
        )

        # Extract target
        y = df[target_col].values if target_col else None

        self._fitted = True
        return X, y

    def transform(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        df = df.copy()
        df = self._handle_missing_values(df)

        X_cat = self._encode_categorical(df[self.categorical_features], fit=False)
        X_num = self._scale_numerical(df[self.numerical_features], fit=False)

        X = (
            np.hstack([X_cat, X_num])
            if X_cat.size > 0 and X_num.size > 0
            else (X_cat if X_cat.size > 0 else X_num)
        )
        y = df[target_col].values if target_col else None

        return X, y

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe."""
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical features."""
        if df.empty or len(self.categorical_features) == 0:
            return np.array([]).reshape(len(df), 0)

        X_cat = df.copy()

        for col in self.categorical_features:
            if col not in X_cat.columns:
                continue

            if fit:
                le = LabelEncoder()
                X_cat[col] = le.fit_transform(X_cat[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                classes_set = set(le.classes_)

                def encode_value(x: Any, encoder: Any = le, valid_classes: set[Any] = classes_set) -> int:
                    return encoder.transform([x])[0] if x in valid_classes else -1

                X_cat[col] = X_cat[col].astype(str).apply(encode_value)

        return X_cat.values.astype(np.float32)

    def _scale_numerical(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale numerical features."""
        if df.empty or len(self.numerical_features) == 0:
            return np.array([]).reshape(len(df), 0)

        X_num = df.values.astype(np.float32)

        if self.scaling_method == "none":
            return X_num

        if fit:
            return self.scaler.fit_transform(X_num).astype(np.float32)
        return self.scaler.transform(X_num).astype(np.float32)


def clean_dataframe(
    df: pd.DataFrame,
    target_col: str,
    drop_na_target: bool = True,
) -> pd.DataFrame:
    """
    Clean dataframe by removing invalid rows.

    Args:
        df: Input dataframe
        target_col: Target column name
        drop_na_target: Whether to drop rows with missing target

    Returns:
        Cleaned dataframe
    """
    df = df.copy()

    if drop_na_target:
        df = df.dropna(subset=[target_col])

    logger.info(f"Cleaned dataframe: {len(df)} samples remaining")
    return df
