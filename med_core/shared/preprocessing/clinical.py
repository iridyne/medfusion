"""Clinical feature preprocessing helpers for case-level models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class ClinicalFeatureTransform:
    """Transformed clinical values plus a missing-value mask."""

    values: np.ndarray
    missing_mask: np.ndarray


class ClinicalFeaturePreprocessor:
    """Fit/transform numeric clinical features with optional missing-mask output."""

    def __init__(
        self,
        *,
        strategy: str = "none",
        normalize: bool = False,
        means: Sequence[float] | None = None,
        stds: Sequence[float] | None = None,
    ) -> None:
        self.strategy = str(strategy)
        self.normalize = bool(normalize)
        self.means = (
            np.asarray(means, dtype=np.float32) if means is not None else None
        )
        self.stds = np.asarray(stds, dtype=np.float32) if stds is not None else None

    @property
    def fitted(self) -> bool:
        return self.means is not None and self.stds is not None

    def fit(self, rows: Sequence[Sequence[float | None]]) -> "ClinicalFeaturePreprocessor":
        values, _missing_mask = self._to_nan_array(rows)
        self.means = np.nanmean(values, axis=0).astype(np.float32)
        self.means = np.nan_to_num(self.means, nan=0.0)

        self.stds = np.nanstd(values, axis=0).astype(np.float32)
        self.stds = np.nan_to_num(self.stds, nan=1.0)
        self.stds[self.stds <= 0] = 1.0
        return self

    def transform(
        self,
        rows: Sequence[Sequence[float | None]],
    ) -> ClinicalFeatureTransform:
        values, missing_mask = self._to_nan_array(rows)
        if self.normalize:
            if not self.fitted:
                raise RuntimeError("ClinicalFeaturePreprocessor must be fitted before transform")
            values = (values - self.means) / self.stds

        values = np.nan_to_num(values, nan=0.0).astype(np.float32)
        if self.strategy == "none":
            missing_mask = np.zeros_like(missing_mask, dtype=np.float32)

        return ClinicalFeatureTransform(values=values, missing_mask=missing_mask)

    def fit_transform(
        self,
        rows: Sequence[Sequence[float | None]],
    ) -> ClinicalFeatureTransform:
        self.fit(rows)
        return self.transform(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "normalize": self.normalize,
            "means": None if self.means is None else self.means.tolist(),
            "stds": None if self.stds is None else self.stds.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClinicalFeaturePreprocessor":
        return cls(
            strategy=str(payload.get("strategy", "none")),
            normalize=bool(payload.get("normalize", False)),
            means=payload.get("means"),
            stds=payload.get("stds"),
        )

    @staticmethod
    def _to_nan_array(
        rows: Sequence[Sequence[float | None]],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(
            [
                [
                    np.nan if value is None else float(value)
                    for value in row
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        missing_mask = np.isnan(values).astype(np.float32)
        return values, missing_mask
