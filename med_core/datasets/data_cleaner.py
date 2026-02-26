"""
Data cleaning utilities for medical datasets.

Provides reusable data cleaning and preprocessing functionality
that can be composed with dataset classes.
"""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """
    Handles data cleaning and preprocessing for medical datasets.

    Separates data cleaning logic from dataset classes to improve
    modularity and testability.

    Example:
        >>> cleaner = DataCleaner(
        ...     numerical_features=["age", "blood_pressure"],
        ...     categorical_features=["gender", "smoking"],
        ...     missing_strategy="fill_mean",
        ...     normalize=True
        ... )
        >>> cleaned_df, tabular_data, scaler = cleaner.clean_and_prepare(df)
    """

    def __init__(
        self,
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        missing_strategy: Literal["drop", "fill_mean", "fill_zero"] = "fill_mean",
        normalize: bool = True,
    ):
        """
        Initialize data cleaner.

        Args:
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            missing_strategy: Strategy for handling missing values
            normalize: Whether to normalize numerical features
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.missing_strategy = missing_strategy
        self.normalize = normalize
        self.scaler: StandardScaler | None = None

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataframe.

        Args:
            df: Input dataframe

        Returns:
            Cleaned dataframe
        """
        df = df.copy()

        if self.missing_strategy == "drop":
            # Drop rows with any missing values in feature columns
            feature_cols = self.numerical_features + self.categorical_features
            df = df.dropna(subset=feature_cols)

        elif self.missing_strategy == "fill_mean":
            # Fill numerical with mean, categorical with mode
            for col in self.numerical_features:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mean())

            for col in self.categorical_features:
                if col in df.columns:
                    mode = df[col].mode()
                    fill_value = mode.iloc[0] if len(mode) > 0 else 0
                    df[col] = df[col].fillna(fill_value)

        elif self.missing_strategy == "fill_zero":
            # Fill all missing with zero
            feature_cols = self.numerical_features + self.categorical_features
            for col in feature_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        return df

    def prepare_tabular_features(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler | None = None,
    ) -> tuple[np.ndarray, list[str], StandardScaler | None]:
        """
        Prepare tabular features from dataframe.

        Handles numerical normalization and categorical encoding.

        Args:
            df: Input dataframe
            scaler: Pre-fitted scaler (for val/test sets)

        Returns:
            Tuple of (feature_array, feature_names, fitted_scaler)
        """
        feature_arrays = []
        feature_names = []

        # Process numerical features
        if self.numerical_features:
            numerical_data = df[self.numerical_features].values.astype(np.float32)

            if self.normalize:
                if scaler is None:
                    self.scaler = StandardScaler()
                    numerical_data = self.scaler.fit_transform(numerical_data)
                else:
                    self.scaler = scaler
                    numerical_data = self.scaler.transform(numerical_data)

            feature_arrays.append(numerical_data)
            feature_names.extend(self.numerical_features)

        # Process categorical features (simple label encoding)
        for cat_col in self.categorical_features:
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

        return tabular_data, feature_names, self.scaler

    def clean_and_prepare(
        self,
        df: pd.DataFrame,
        scaler: StandardScaler | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str], StandardScaler | None]:
        """
        Clean and prepare data in one call.

        Args:
            df: Input dataframe
            scaler: Pre-fitted scaler (for val/test sets)

        Returns:
            Tuple of (cleaned_df, tabular_data, feature_names, fitted_scaler)
        """
        cleaned_df = self.handle_missing_values(df)
        tabular_data, feature_names, fitted_scaler = self.prepare_tabular_features(
            cleaned_df, scaler,
        )
        return cleaned_df, tabular_data, feature_names, fitted_scaler
