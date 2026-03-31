"""Three-phase CT dataset for the SMuRF-Lite MVI demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from med_core.shared.data_utils import load_dicom_series
from med_core.shared.preprocessing.clinical import ClinicalFeaturePreprocessor


@dataclass(slots=True)
class ThreePhaseCTCaseRecord:
    case_id: str
    arterial_series_dir: str
    portal_series_dir: str
    noncontrast_series_dir: str
    mvi_binary: int
    clinical_features: list[float | None]


class ThreePhaseCTCaseDataset(Dataset[dict[str, Any]]):
    """Load one three-phase CT volume plus tabular features per sample."""

    def __init__(
        self,
        records: list[ThreePhaseCTCaseRecord],
        target_shape: tuple[int, int, int],
        window_preset: str = "liver",
        clinical_preprocessor: ClinicalFeaturePreprocessor | None = None,
    ) -> None:
        self.records = records
        self.target_shape = target_shape
        self.window_preset = window_preset
        self.clinical_preprocessor = clinical_preprocessor
        (
            self._clinical_values,
            self._clinical_missing_masks,
        ) = self._prepare_clinical_features()

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, object]],
        target_shape: tuple[int, int, int],
        window_preset: str = "liver",
        clinical_preprocessor: ClinicalFeaturePreprocessor | None = None,
    ) -> "ThreePhaseCTCaseDataset":
        def _resolve_series_dir(record: dict[str, object], *keys: str) -> str:
            for key in keys:
                value = record.get(key)
                if value:
                    return str(value)
            raise KeyError(keys[0])

        normalized = [
            ThreePhaseCTCaseRecord(
                case_id=str(record["case_id"]),
                arterial_series_dir=str(record["arterial_series_dir"]),
                portal_series_dir=str(record["portal_series_dir"]),
                noncontrast_series_dir=_resolve_series_dir(
                    record,
                    "noncontrast_series_dir",
                    "delayed_series_dir",
                    "nc_series_dir",
                ),
                mvi_binary=int(record["mvi_binary"]),
                clinical_features=[
                    None if value is None else float(value)
                    for value in list(record["clinical_features"])
                ],
            )
            for record in records
        ]
        return cls(
            records=normalized,
            target_shape=target_shape,
            window_preset=window_preset,
            clinical_preprocessor=clinical_preprocessor,
        )

    @classmethod
    def from_manifest_dataframe(
        cls,
        dataframe: pd.DataFrame,
        *,
        phase_dir_columns: dict[str, str],
        clinical_feature_columns: list[str],
        target_column: str,
        patient_id_column: str,
        target_shape: tuple[int, int, int],
        window_preset: str = "liver",
        clinical_preprocessor: ClinicalFeaturePreprocessor | None = None,
    ) -> "ThreePhaseCTCaseDataset":
        records: list[dict[str, object]] = []
        for row in dataframe.to_dict(orient="records"):
            records.append(
                {
                    "case_id": row[patient_id_column],
                    "arterial_series_dir": row[phase_dir_columns["arterial"]],
                    "portal_series_dir": row[phase_dir_columns["portal"]],
                    "noncontrast_series_dir": row[phase_dir_columns["noncontrast"]],
                    "mvi_binary": row[target_column],
                    "clinical_features": [
                        None if pd.isna(row[column]) else float(row[column])
                        for column in clinical_feature_columns
                    ],
                }
            )
        return cls.from_records(
            records=records,
            target_shape=target_shape,
            window_preset=window_preset,
            clinical_preprocessor=clinical_preprocessor,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "case_id": record.case_id,
            "arterial": self._load_phase(record.arterial_series_dir),
            "portal": self._load_phase(record.portal_series_dir),
            "noncontrast": self._load_phase(record.noncontrast_series_dir),
            "clinical": torch.tensor(self._clinical_values[index], dtype=torch.float32),
            "clinical_missing_mask": torch.tensor(
                self._clinical_missing_masks[index],
                dtype=torch.float32,
            ),
            "label": torch.tensor(record.mvi_binary, dtype=torch.long),
        }

    def get_phase_render_context(self, index: int, phase_name: str) -> dict[str, Any]:
        """Return native and model-space context for postprocessing visualizations."""
        record = self.records[index]
        series_dir = self._get_phase_series_dir(record, phase_name)
        original_volume = self._load_phase_volume(series_dir)
        return {
            "case_id": record.case_id,
            "phase": phase_name,
            "original_volume": original_volume,
            "original_shape": tuple(int(value) for value in original_volume.shape),
            "model_shape": tuple(int(value) for value in self.target_shape),
        }

    def _prepare_clinical_features(self) -> tuple[list[list[float]], list[list[float]]]:
        raw_rows = [record.clinical_features for record in self.records]
        preprocessor = self.clinical_preprocessor or ClinicalFeaturePreprocessor(
            strategy="zero_with_mask",
            normalize=False,
        )
        transformed = preprocessor.transform(raw_rows)
        return transformed.values.tolist(), transformed.missing_mask.tolist()

    def _get_phase_series_dir(self, record: ThreePhaseCTCaseRecord, phase_name: str) -> str:
        phase_mapping = {
            "arterial": record.arterial_series_dir,
            "portal": record.portal_series_dir,
            "noncontrast": record.noncontrast_series_dir,
        }
        try:
            return phase_mapping[phase_name]
        except KeyError as error:
            raise ValueError(f"Unsupported phase name: {phase_name}") from error

    def _load_phase_volume(self, series_dir: str) -> np.ndarray:
        if not Path(series_dir).exists():
            raise FileNotFoundError(f"DICOM series directory does not exist: {series_dir}")
        return load_dicom_series(
            directory=series_dir,
            window_preset=self.window_preset,
            sort_by="instance",
        )

    def _load_phase(self, series_dir: str) -> torch.Tensor:
        volume = self._load_phase_volume(series_dir)
        tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if list(volume.shape) != list(self.target_shape):
            tensor = F.interpolate(
                tensor,
                size=self.target_shape,
                mode="trilinear",
                align_corners=False,
            )

        return tensor.squeeze(0)
