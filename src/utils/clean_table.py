"""
Clinical Data Sanitizer Module

This module provides utilities for cleaning and preprocessing clinical tabular data.
It handles common data quality issues in medical datasets:
- Outlier detection and handling
- Missing value imputation
- Categorical encoding (one-hot, label encoding)
- Logical anomaly detection (e.g., age=200, BP=0)
- Integration with data_dictionary.yaml for validation

Key Features:
- Automatic outlier detection based on physiological ranges
- Smart categorical encoding with medical domain knowledge
- Comprehensive logging of data quality issues
- Type-safe implementation with Python 3.12+ hints

Usage:
    from src.utils.clean_table import ClinicalDataSanitizer

    sanitizer = ClinicalDataSanitizer("docs/data_dictionary.yaml")
    df_clean = sanitizer.clean(df_raw)
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml


class ClinicalDataSanitizer:
    """
    Clinical data sanitizer for medical tabular data.

    This class provides comprehensive data cleaning for clinical features,
    including outlier detection, categorical encoding, and logical validation.
    """

    def __init__(
        self,
        dictionary_path: str | Path | None = None,
        outlier_method: Literal["iqr", "zscore", "range"] = "range",
        missing_strategy: Literal[
            "drop", "mean", "median", "mode", "forward_fill"
        ] = "median",
        encoding_strategy: Literal["onehot", "label", "ordinal"] = "label",
        verbose: bool = True,
    ):
        """
        Initialize clinical data sanitizer.

        Args:
            dictionary_path: Path to data_dictionary.yaml (optional)
            outlier_method: Method for outlier detection
            missing_strategy: Strategy for handling missing values
            encoding_strategy: Strategy for categorical encoding
            verbose: Whether to print cleaning progress
        """
        self.dictionary_path = Path(dictionary_path) if dictionary_path else None
        self.outlier_method = outlier_method
        self.missing_strategy = missing_strategy
        self.encoding_strategy = encoding_strategy
        self.verbose = verbose

        # Load data dictionary if provided
        self.dictionary = self._load_dictionary() if self.dictionary_path else {}

        # Statistics tracking
        self.cleaning_stats = {
            "outliers_detected": 0,
            "outliers_fixed": 0,
            "missing_values": 0,
            "missing_filled": 0,
            "categorical_encoded": 0,
            "logical_errors": 0,
        }

    def _load_dictionary(self) -> dict:
        """Load data dictionary from YAML file."""
        if not self.dictionary_path.exists():
            if self.verbose:
                print(f"Warning: Dictionary not found at {self.dictionary_path}")
            return {}

        with open(self.dictionary_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_valid_range(self, column: str) -> tuple[float, float] | None:
        """Get valid range for a column from data dictionary."""
        if not self.dictionary:
            return None

        # Search through all categories
        for category_data in self.dictionary.values():
            if isinstance(category_data, dict):
                for field_name, field_info in category_data.items():
                    if isinstance(field_info, dict):
                        # Check if this field matches the column
                        if field_name == column or column in field_info.get(
                            "aliases", []
                        ):
                            return field_info.get("valid_range")

        return None

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        """
        Detect outliers in a numeric column.

        Args:
            df: Input dataframe
            column: Column name

        Returns:
            Boolean series indicating outliers (True = outlier)
        """
        if column not in df.columns:
            return pd.Series([False] * len(df), index=df.index)

        series = df[column]

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(series):
            return pd.Series([False] * len(df), index=df.index)

        # Method 1: Range-based (using data dictionary)
        if self.outlier_method == "range":
            valid_range = self._get_valid_range(column)
            if valid_range:
                min_val, max_val = valid_range
                outliers = (series < min_val) | (series > max_val)
                return outliers

        # Method 2: IQR-based
        elif self.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
            return outliers

        # Method 3: Z-score based
        elif self.outlier_method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > 4
            return outliers

        return pd.Series([False] * len(df), index=df.index)

    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        strategy: Literal["clip", "nan", "remove"] = "nan",
    ) -> pd.DataFrame:
        """
        Handle outliers in a column.

        Args:
            df: Input dataframe
            column: Column name
            strategy: How to handle outliers
                - "clip": Clip to valid range
                - "nan": Replace with NaN
                - "remove": Remove rows with outliers

        Returns:
            Cleaned dataframe
        """
        outliers = self.detect_outliers(df, column)
        num_outliers = outliers.sum()

        if num_outliers == 0:
            return df

        self.cleaning_stats["outliers_detected"] += num_outliers

        if self.verbose:
            print(f"  Found {num_outliers} outliers in '{column}'")

        df_clean = df.copy()

        if strategy == "nan":
            df_clean.loc[outliers, column] = np.nan
            self.cleaning_stats["outliers_fixed"] += num_outliers

        elif strategy == "clip":
            valid_range = self._get_valid_range(column)
            if valid_range:
                min_val, max_val = valid_range
                df_clean[column] = df_clean[column].clip(min_val, max_val)
                self.cleaning_stats["outliers_fixed"] += num_outliers

        elif strategy == "remove":
            df_clean = df_clean[~outliers]
            self.cleaning_stats["outliers_fixed"] += num_outliers

        return df_clean

    def detect_logical_errors(self, df: pd.DataFrame) -> dict[str, list[int]]:
        """
        Detect logical errors in clinical data.

        Examples:
        - Age = 0 or > 120
        - Blood pressure = 0
        - Negative values for positive-only measurements

        Args:
            df: Input dataframe

        Returns:
            Dictionary mapping column names to lists of error indices
        """
        errors = {}

        # Common logical checks
        checks = {
            "age": lambda x: (x <= 0) | (x > 120),
            "systolic_bp": lambda x: (x <= 0) | (x > 300),
            "diastolic_bp": lambda x: (x <= 0) | (x > 200),
            "heart_rate": lambda x: (x <= 0) | (x > 250),
            "temperature": lambda x: (x < 30) | (x > 45),
            "weight": lambda x: (x <= 0) | (x > 500),
            "height": lambda x: (x <= 0) | (x > 300),
            "bmi": lambda x: (x <= 0) | (x > 100),
        }

        for col in df.columns:
            col_lower = col.lower()

            # Check if column matches any known field
            for field, check_func in checks.items():
                if field in col_lower:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        error_mask = check_func(df[col])
                        error_indices = df[error_mask].index.tolist()
                        if error_indices:
                            errors[col] = error_indices
                            self.cleaning_stats["logical_errors"] += len(error_indices)

        return errors

    def encode_categorical(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.DataFrame:
        """
        Encode categorical column to numeric.

        Args:
            df: Input dataframe
            column: Column name

        Returns:
            Dataframe with encoded column
        """
        if column not in df.columns:
            return df

        df_encoded = df.copy()

        # Common medical categorical mappings
        gender_map = {
            "male": 0,
            "男": 0,
            "M": 0,
            "m": 0,
            "female": 1,
            "女": 1,
            "F": 1,
            "f": 1,
        }

        smoking_map = {
            "never": 0,
            "从不": 0,
            "no": 0,
            "否": 0,
            "former": 1,
            "曾经": 1,
            "quit": 1,
            "current": 2,
            "目前": 2,
            "yes": 2,
            "是": 2,
        }

        binary_map = {
            "no": 0,
            "否": 0,
            "false": 0,
            "0": 0,
            "yes": 1,
            "是": 1,
            "true": 1,
            "1": 1,
        }

        col_lower = column.lower()

        # Apply appropriate mapping
        if "gender" in col_lower or "sex" in col_lower or "性别" in col_lower:
            df_encoded[column] = df_encoded[column].astype(str).map(gender_map)
            self.cleaning_stats["categorical_encoded"] += 1

        elif "smok" in col_lower or "吸烟" in col_lower:
            df_encoded[column] = df_encoded[column].astype(str).map(smoking_map)
            self.cleaning_stats["categorical_encoded"] += 1

        elif df[column].nunique() == 2:
            # Binary categorical
            df_encoded[column] = df_encoded[column].astype(str).map(binary_map)
            if df_encoded[column].isna().all():
                # Fallback: simple label encoding
                df_encoded[column] = pd.Categorical(df[column]).codes
            self.cleaning_stats["categorical_encoded"] += 1

        elif self.encoding_strategy == "label":
            # General label encoding
            df_encoded[column] = pd.Categorical(df[column]).codes
            self.cleaning_stats["categorical_encoded"] += 1

        return df_encoded

    def handle_missing(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> pd.DataFrame:
        """
        Handle missing values in a column.

        Args:
            df: Input dataframe
            column: Column name

        Returns:
            Dataframe with missing values handled
        """
        if column not in df.columns:
            return df

        missing_count = df[column].isna().sum()
        if missing_count == 0:
            return df

        self.cleaning_stats["missing_values"] += missing_count

        df_filled = df.copy()

        if self.missing_strategy == "drop":
            df_filled = df_filled.dropna(subset=[column])

        elif self.missing_strategy == "mean":
            if pd.api.types.is_numeric_dtype(df[column]):
                df_filled[column].fillna(df[column].mean(), inplace=True)
                self.cleaning_stats["missing_filled"] += missing_count

        elif self.missing_strategy == "median":
            if pd.api.types.is_numeric_dtype(df[column]):
                df_filled[column].fillna(df[column].median(), inplace=True)
                self.cleaning_stats["missing_filled"] += missing_count

        elif self.missing_strategy == "mode":
            df_filled[column].fillna(df[column].mode()[0], inplace=True)
            self.cleaning_stats["missing_filled"] += missing_count

        elif self.missing_strategy == "forward_fill":
            df_filled[column].fillna(method="ffill", inplace=True)
            self.cleaning_stats["missing_filled"] += missing_count

        return df_filled

    def clean(
        self,
        df: pd.DataFrame,
        outlier_strategy: Literal["clip", "nan", "remove"] = "nan",
    ) -> pd.DataFrame:
        """
        Perform comprehensive cleaning on clinical dataframe.

        Args:
            df: Input dataframe
            outlier_strategy: How to handle outliers

        Returns:
            Cleaned dataframe
        """
        if self.verbose:
            print("Starting clinical data cleaning...")
            print(f"Input shape: {df.shape}")

        df_clean = df.copy()

        # Reset statistics
        self.cleaning_stats = dict.fromkeys(self.cleaning_stats, 0)

        # Step 1: Detect logical errors
        if self.verbose:
            print("\nStep 1: Detecting logical errors...")

        logical_errors = self.detect_logical_errors(df_clean)
        if logical_errors and self.verbose:
            for col, indices in logical_errors.items():
                print(f"  Logical errors in '{col}': {len(indices)} rows")

        # Step 2: Handle outliers
        if self.verbose:
            print("\nStep 2: Handling outliers...")

        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean = self.handle_outliers(df_clean, col, outlier_strategy)

        # Step 3: Encode categorical variables
        if self.verbose:
            print("\nStep 3: Encoding categorical variables...")

        for col in df_clean.columns:
            if (
                df_clean[col].dtype == "object"
                or df_clean[col].dtype.name == "category"
            ):
                df_clean = self.encode_categorical(df_clean, col)

        # Step 4: Handle missing values
        if self.verbose:
            print("\nStep 4: Handling missing values...")

        for col in df_clean.columns:
            df_clean = self.handle_missing(df_clean, col)

        # Summary
        if self.verbose:
            print("\n" + "=" * 60)
            print("Cleaning Summary:")
            print(f"  Outliers detected: {self.cleaning_stats['outliers_detected']}")
            print(f"  Outliers fixed: {self.cleaning_stats['outliers_fixed']}")
            print(f"  Logical errors: {self.cleaning_stats['logical_errors']}")
            print(f"  Missing values: {self.cleaning_stats['missing_values']}")
            print(f"  Missing filled: {self.cleaning_stats['missing_filled']}")
            print(
                f"  Categorical encoded: {self.cleaning_stats['categorical_encoded']}"
            )
            print(f"  Output shape: {df_clean.shape}")
            print("=" * 60)

        return df_clean

    def get_cleaning_report(self) -> dict:
        """Get detailed cleaning statistics."""
        return self.cleaning_stats.copy()


# Example usage
if __name__ == "__main__":
    # Create sample clinical data with issues
    sample_data = {
        "patient_id": ["P001", "P002", "P003", "P004", "P005"],
        "age": [45, 200, 38, -5, 61],  # Outliers: 200, -5
        "gender": ["男", "F", "女", "M", "男"],
        "systolic_bp": [120, 0, 145, 160, 135],  # Outlier: 0
        "diastolic_bp": [80, 70, 95, 100, 85],
        "glucose": [5.2, np.nan, 7.8, 9.1, 5.0],  # Missing value
        "smoking": ["从不", "yes", "no", "目前", "former"],
    }

    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)
    print("\n")

    # Initialize sanitizer
    sanitizer = ClinicalDataSanitizer(
        dictionary_path="docs/data_dictionary.yaml",
        outlier_method="range",
        missing_strategy="median",
        encoding_strategy="label",
        verbose=True,
    )

    # Clean data
    df_clean = sanitizer.clean(df, outlier_strategy="nan")

    print("\nCleaned DataFrame:")
    print(df_clean)

    # Get report
    report = sanitizer.get_cleaning_report()
    print("\nCleaning Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
