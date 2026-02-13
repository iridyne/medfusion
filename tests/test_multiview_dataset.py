"""
Tests for multi-view dataset functionality.

Tests:
- MedicalMultiViewDataset creation
- from_csv_multiview factory method
- View availability handling
- Missing view strategies
- Multi-view data loading
"""

import pytest
import torch

from med_core.datasets import MedicalMultiViewDataset
from med_core.datasets.multiview_types import MultiViewConfig


class TestMultiViewDataset:
    """Test multi-view dataset functionality."""

    def test_from_csv_multiview_basic(self, sample_multiview_csv_data):
        """Test basic multi-view dataset creation from CSV."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, scaler = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            categorical_features=["gender"],
            view_config=config,
        )

        assert len(dataset) == sample_multiview_csv_data["num_samples"]
        assert scaler is not None

    def test_multiview_dataset_getitem(self, sample_multiview_csv_data):
        """Test __getitem__ returns correct format."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        sample = dataset[0]
        assert isinstance(sample, tuple)
        assert len(sample) == 3

        views, tabular, label = sample
        assert isinstance(views, dict)
        assert "axial" in views
        assert "coronal" in views
        assert "sagittal" in views
        assert isinstance(tabular, torch.Tensor)
        assert isinstance(label, (int, torch.Tensor))

    def test_multiview_view_statistics(self, sample_multiview_csv_data):
        """Test view availability statistics."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        stats = dataset.get_statistics()
        assert "view_availability" in stats
        assert "total_samples" in stats
        assert stats["total_samples"] == len(dataset)

    def test_multiview_with_missing_views(self, sample_multiview_csv_data, temp_dir):
        """Test handling of missing views."""
        import pandas as pd

        # Create CSV with some missing views
        df = pd.read_csv(sample_multiview_csv_data["csv_path"])
        # Set some view paths to NaN
        df.loc[0, "coronal_path"] = None
        df.loc[1, "sagittal_path"] = None

        csv_path = temp_dir / "missing_views.csv"
        df.to_csv(csv_path, index=False)

        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=["axial"],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=csv_path,
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        # Should still load samples with at least the required view
        assert len(dataset) > 0

    def test_multiview_required_views(self, sample_multiview_csv_data):
        """Test required views validation."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=["axial", "coronal"],
            handle_missing="skip",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        # All samples should have required views
        for i in range(len(dataset)):
            views, _, _ = dataset[i]
            assert "axial" in views
            assert "coronal" in views

    def test_multiview_handle_missing_zero(self, sample_multiview_csv_data, temp_dir):
        """Test handle_missing='zero' strategy."""
        import pandas as pd

        # Create CSV with missing view
        df = pd.read_csv(sample_multiview_csv_data["csv_path"])
        df.loc[0, "coronal_path"] = None
        csv_path = temp_dir / "missing_zero.csv"
        df.to_csv(csv_path, index=False)

        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=csv_path,
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        views, _, _ = dataset[0]
        # Missing view should be replaced with zeros
        assert "coronal" in views
        assert torch.all(views["coronal"] == 0)

    def test_multiview_handle_missing_duplicate(self, sample_multiview_csv_data, temp_dir):
        """Test handle_missing='duplicate' strategy."""
        import pandas as pd

        # Create CSV with missing view
        df = pd.read_csv(sample_multiview_csv_data["csv_path"])
        df.loc[0, "coronal_path"] = None
        csv_path = temp_dir / "missing_dup.csv"
        df.to_csv(csv_path, index=False)

        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="duplicate",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=csv_path,
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
        )

        views, _, _ = dataset[0]
        # Missing view should be duplicated from another view
        assert "coronal" in views
        assert not torch.all(views["coronal"] == 0)

    def test_multiview_feature_names(self, sample_multiview_csv_data):
        """Test feature names are preserved."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            categorical_features=["gender"],
            view_config=config,
        )

        feature_names = dataset.get_feature_names()
        assert "age" in feature_names
        assert "bmi" in feature_names
        assert "gender" in feature_names

    def test_multiview_patient_ids(self, sample_multiview_csv_data):
        """Test patient IDs are preserved."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            patient_id_column="patient_id",
            view_config=config,
        )

        patient_id = dataset.get_patient_id(0)
        assert patient_id is not None
        assert patient_id.startswith("P")

    def test_multiview_with_transforms(self, sample_multiview_csv_data):
        """Test dataset with image transforms."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        dataset, _ = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
            transform=transform,
        )

        views, _, _ = dataset[0]
        # Check transformed size
        for view_tensor in views.values():
            if not torch.all(view_tensor == 0):  # Skip zero-filled views
                assert view_tensor.shape[-2:] == (128, 128)

    def test_multiview_scaler_reuse(self, sample_multiview_csv_data):
        """Test scaler can be reused for validation/test sets."""
        config = MultiViewConfig(
            view_names=["axial", "coronal", "sagittal"],
            required_views=[],
            handle_missing="zero",
        )

        # Create train dataset
        train_dataset, train_scaler = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
            normalize_features=True,
        )

        # Create val dataset with same scaler
        val_dataset, val_scaler = MedicalMultiViewDataset.from_csv_multiview(
            csv_path=sample_multiview_csv_data["csv_path"],
            image_dir=sample_multiview_csv_data["image_dir"],
            view_columns=sample_multiview_csv_data["view_columns"],
            target_column="label",
            numerical_features=["age", "bmi"],
            view_config=config,
            normalize_features=True,
            scaler=train_scaler,
        )

        assert val_scaler is train_scaler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
