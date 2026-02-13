"""
Medical Data Normalizer Module

This module provides utilities for normalizing medical data using the data dictionary.
It automatically maps field names, validates data types, and handles missing values.

Key Features:
- Automatic field name normalization using data_dictionary.yaml
- Data type validation and conversion
- Range validation for numeric fields
- Missing value handling
- Type-safe implementation with Python 3.12+ hints

Usage:
    from src.data.data_normalizer import MedicalDataNormalizer

    normalizer = MedicalDataNormalizer("docs/data_dictionary.yaml")
    df_normalized = normalizer.normalize_dataframe(df_raw)
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml


class MedicalDataNormalizer:
    """
    Medical data normalizer using data dictionary for field mapping.

    This class loads a data dictionary YAML file and provides methods to:
    - Map alternative field names to standard names
    - Validate and convert data types
    - Check value ranges
    - Handle missing values
    """

    def __init__(self, dictionary_path: str | Path):
        """
        Initialize the normalizer with a data dictionary.

        Args:
            dictionary_path: Path to data_dictionary.yaml file
        """
        self.dictionary_path = Path(dictionary_path)
        self.dictionary = self._load_dictionary()
        self.field_mapping = self._build_field_mapping()

    def _load_dictionary(self) -> dict:
        """Load data dictionary from YAML file."""
        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                f"Data dictionary not found: {self.dictionary_path}"
            )

        with open(self.dictionary_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_field_mapping(self) -> dict[str, str]:
        """
        Build a mapping from all aliases to standard field names.

        Returns:
            Dictionary mapping alias -> standard_name
        """
        mapping = {}

        # Iterate through all categories in the dictionary
        for category_name, category_data in self.dictionary.items():
            if category_name in ["version", "last_updated"]:
                continue

            if not isinstance(category_data, dict):
                continue

            # Iterate through fields in each category
            for standard_name, field_info in category_data.items():
                if not isinstance(field_info, dict):
                    continue

                # Add standard name to itself
                mapping[standard_name] = standard_name

                # Add all aliases
                aliases = field_info.get("aliases", [])
                for alias in aliases:
                    mapping[alias] = standard_name

        return mapping

    def normalize_field_name(self, field_name: str) -> str:
        """
        Normalize a field name to its standard form.

        Args:
            field_name: Original field name

        Returns:
            Standard field name

        Raises:
            ValueError: If field name is not found in dictionary
        """
        if field_name in self.field_mapping:
            return self.field_mapping[field_name]

        # Try case-insensitive matching
        field_lower = field_name.lower()
        for alias, standard in self.field_mapping.items():
            if alias.lower() == field_lower:
                return standard

        raise ValueError(f"Unknown field name: {field_name}")

    def get_field_info(self, standard_name: str) -> dict | None:
        """
        Get field information from the dictionary.

        Args:
            standard_name: Standard field name

        Returns:
            Field information dictionary or None if not found
        """
        for category_data in self.dictionary.values():
            if isinstance(category_data, dict) and standard_name in category_data:
                return category_data[standard_name]
        return None

    def validate_value(
        self, value: Any, field_name: str, raise_error: bool = False
    ) -> tuple[bool, str | None]:
        """
        Validate a value against field constraints.

        Args:
            value: Value to validate
            field_name: Standard field name
            raise_error: Whether to raise error on validation failure

        Returns:
            Tuple of (is_valid, error_message)
        """
        if pd.isna(value):
            return True, None

        field_info = self.get_field_info(field_name)
        if field_info is None:
            return True, None

        # Check numeric range
        if "valid_range" in field_info:
            try:
                value_float = float(value)
                min_val, max_val = field_info["valid_range"]
                if not (min_val <= value_float <= max_val):
                    msg = f"{field_name} value {value} outside valid range [{min_val}, {max_val}]"
                    if raise_error:
                        raise ValueError(msg)
                    return False, msg
            except (ValueError, TypeError):
                pass

        # Check categorical values
        if "valid_values" in field_info:
            valid_values = field_info["valid_values"]
            if value not in valid_values:
                # Try string conversion
                if str(value) not in [str(v) for v in valid_values]:
                    msg = f"{field_name} value {value} not in valid values: {valid_values}"
                    if raise_error:
                        raise ValueError(msg)
                    return False, msg

        return True, None

    def convert_value(self, value: Any, field_name: str) -> Any:
        """
        Convert value to appropriate type based on field definition.

        Args:
            value: Value to convert
            field_name: Standard field name

        Returns:
            Converted value
        """
        if pd.isna(value):
            return np.nan

        field_info = self.get_field_info(field_name)
        if field_info is None:
            return value

        field_type = field_info.get("type", "text")

        try:
            if field_type == "numeric":
                return float(value)

            elif field_type == "binary":
                # Handle various binary representations
                if isinstance(value, (bool, int, float)):
                    return int(bool(value))

                value_str = str(value).lower()
                if value_str in ["yes", "true", "是", "1", "1.0"]:
                    return 1
                elif value_str in ["no", "false", "否", "0", "0.0"]:
                    return 0
                else:
                    return int(value)

            elif field_type == "categorical":
                # Handle categorical encoding if defined
                if "encoding" in field_info:
                    encoding = field_info["encoding"]
                    for standard_value, aliases in encoding.items():
                        if value in aliases or str(value) in [str(a) for a in aliases]:
                            return standard_value
                return str(value)

            elif field_type == "text":
                return str(value)

            else:
                return value

        except (ValueError, TypeError):
            return value

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        validate: bool = True,
        handle_missing: Literal["drop", "fill_mean", "fill_median", "keep"] = "keep",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize a complete dataframe using the data dictionary.

        Args:
            df: Input dataframe
            validate: Whether to validate values
            handle_missing: How to handle missing values
            verbose: Whether to print progress information

        Returns:
            Normalized dataframe with standard field names
        """
        df_normalized = df.copy()

        # Step 1: Normalize column names
        if verbose:
            print("Step 1: Normalizing column names...")

        column_mapping = {}
        unmapped_columns = []

        for col in df.columns:
            try:
                standard_name = self.normalize_field_name(col)
                column_mapping[col] = standard_name
            except ValueError:
                unmapped_columns.append(col)

        if unmapped_columns and verbose:
            print(
                f"  Warning: {len(unmapped_columns)} columns not found in dictionary:"
            )
            for col in unmapped_columns[:5]:
                print(f"    - {col}")
            if len(unmapped_columns) > 5:
                print(f"    ... and {len(unmapped_columns) - 5} more")

        df_normalized = df_normalized.rename(columns=column_mapping)

        # Step 2: Convert data types
        if verbose:
            print("Step 2: Converting data types...")

        for col in df_normalized.columns:
            if col in unmapped_columns:
                continue

            df_normalized[col] = df_normalized[col].apply(
                lambda x, c=col: self.convert_value(x, c)
            )

        # Step 3: Validate values
        if validate and verbose:
            print("Step 3: Validating values...")

        validation_errors = []

        if validate:
            for col in df_normalized.columns:
                if col in unmapped_columns:
                    continue

                for idx, value in df_normalized[col].items():
                    is_valid, error_msg = self.validate_value(value, col)
                    if not is_valid:
                        validation_errors.append((idx, col, error_msg))

        if validation_errors and verbose:
            print(f"  Found {len(validation_errors)} validation errors")
            for idx, col, msg in validation_errors[:5]:
                print(f"    Row {idx}, {col}: {msg}")
            if len(validation_errors) > 5:
                print(f"    ... and {len(validation_errors) - 5} more")

        # Step 4: Handle missing values
        if verbose:
            print(f"Step 4: Handling missing values (strategy: {handle_missing})...")

        if handle_missing == "drop":
            df_normalized = df_normalized.dropna()

        elif handle_missing == "fill_mean":
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
            df_normalized[numeric_cols] = df_normalized[numeric_cols].fillna(
                df_normalized[numeric_cols].mean()
            )

        elif handle_missing == "fill_median":
            numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
            df_normalized[numeric_cols] = df_normalized[numeric_cols].fillna(
                df_normalized[numeric_cols].median()
            )

        # keep: do nothing

        if verbose:
            print("\nNormalization complete!")
            print(f"  Input shape: {df.shape}")
            print(f"  Output shape: {df_normalized.shape}")
            print(f"  Mapped columns: {len(column_mapping)}")
            print(f"  Unmapped columns: {len(unmapped_columns)}")

        return df_normalized

    def get_statistics(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Get statistics for normalized dataframe.

        Args:
            df: Normalized dataframe

        Returns:
            Dictionary of statistics for each field
        """
        stats = {}

        for col in df.columns:
            field_info = self.get_field_info(col)
            if field_info is None:
                continue

            col_stats = {
                "type": field_info.get("type", "unknown"),
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
            }

            if field_info.get("type") == "numeric":
                col_stats.update(
                    {
                        "mean": float(df[col].mean())
                        if not df[col].isna().all()
                        else None,
                        "std": float(df[col].std())
                        if not df[col].isna().all()
                        else None,
                        "min": float(df[col].min())
                        if not df[col].isna().all()
                        else None,
                        "max": float(df[col].max())
                        if not df[col].isna().all()
                        else None,
                    }
                )

            elif field_info.get("type") in ["categorical", "binary"]:
                value_counts = df[col].value_counts().to_dict()
                col_stats["value_counts"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }

            stats[col] = col_stats

        return stats


# Example usage
if __name__ == "__main__":
    # Example: Load and normalize a medical dataset

    # Initialize normalizer
    normalizer = MedicalDataNormalizer("docs/data_dictionary.yaml")

    # Create sample data with various field name formats
    sample_data = {
        "年龄": [45, 52, 38, 61, 29],
        "Gender": ["男", "F", "女", "M", "男"],
        "BMI": [24.5, 28.3, 22.1, 31.2, 19.8],
        "收缩压": [120, 145, 110, 160, 105],
        "DBP": [80, 95, 70, 100, 68],
        "血糖": [5.2, 7.8, 4.9, 9.1, 5.0],
        "Hypertension": ["否", "yes", "no", "是", "no"],
    }

    df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(df)
    print("\n" + "=" * 80 + "\n")

    # Normalize the dataframe
    df_normalized = normalizer.normalize_dataframe(
        df, validate=True, handle_missing="keep", verbose=True
    )

    print("\n" + "=" * 80 + "\n")
    print("Normalized DataFrame:")
    print(df_normalized)

    print("\n" + "=" * 80 + "\n")
    print("Statistics:")
    stats = normalizer.get_statistics(df_normalized)
    for field, field_stats in stats.items():
        print(f"\n{field}:")
        for key, value in field_stats.items():
            print(f"  {key}: {value}")
