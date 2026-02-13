"""
Dataset module for medical multimodal learning.

Provides:
- Base dataset classes for multimodal data (single and multi-view)
- Medical image + tabular data loaders
- Data preprocessing and augmentation utilities
- Multi-view support for multiple images per patient
"""

from med_core.datasets.base import BaseMultimodalDataset
from med_core.datasets.data_cleaner import DataCleaner
from med_core.datasets.medical import (
    MedicalMultimodalDataset,
    create_dataloaders,
    split_dataset,
)
from med_core.datasets.medical_multiview import MedicalMultiViewDataset
from med_core.datasets.multiview_base import BaseMultiViewDataset
from med_core.datasets.multiview_types import (
    MultiViewConfig,
    ViewDict,
    ViewTensor,
    convert_to_multiview_paths,
    create_single_view_dict,
)
from med_core.datasets.transforms import (
    get_medical_augmentation,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    # Single-view (legacy)
    "BaseMultimodalDataset",
    "MedicalMultimodalDataset",
    # Multi-view
    "BaseMultiViewDataset",
    "MedicalMultiViewDataset",
    "MultiViewConfig",
    "ViewDict",
    "ViewTensor",
    "create_single_view_dict",
    "convert_to_multiview_paths",
    # Data cleaning
    "DataCleaner",
    # Data loading utilities
    "create_dataloaders",
    "split_dataset",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "get_medical_augmentation",
]
