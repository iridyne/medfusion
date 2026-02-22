"""
Tests for dataset loading and splitting functionality.

Tests cover:
- BaseMultimodalDataset core functionality
- MedicalMultimodalDataset loading from CSV
- Dataset splitting (train/val/test)
- Data preprocessing and transformations
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from med_core.datasets import (
    MedicalMultimodalDataset,
    create_dataloaders,
    split_dataset,
)


class TestBaseMultimodalDataset(unittest.TestCase):
    """Test BaseMultimodalDataset functionality."""

    def setUp(self):
        """Create temporary test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.temp_dir.name) / "images"
        self.image_dir.mkdir()

        # Create synthetic images
        self.num_samples = 20
        self.image_paths = []
        for i in range(self.num_samples):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img_path = self.image_dir / f"sample_{i:03d}.png"
            img.save(img_path)
            self.image_paths.append(str(img_path))

        # Create synthetic tabular data
        self.tabular_data = np.random.randn(self.num_samples, 5).astype(np.float32)
        self.labels = np.random.randint(0, 2, self.num_samples)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        dataset = MedicalMultimodalDataset(
            image_paths=self.image_paths,
            tabular_data=self.tabular_data,
            labels=self.labels,
        )
        self.assertEqual(len(dataset), self.num_samples)
        self.assertEqual(dataset.get_tabular_dim(), 5)

    def test_dataset_getitem(self):
        """Test dataset indexing."""
        dataset = MedicalMultimodalDataset(
            image_paths=self.image_paths,
            tabular_data=self.tabular_data,
            labels=self.labels,
        )
        image, tabular, label = dataset[0]
        self.assertIsInstance(image, Image.Image)
        self.assertIsInstance(tabular, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(tabular.shape, (5,))

    def test_class_distribution(self):
        """Test class distribution calculation."""
        dataset = MedicalMultimodalDataset(
            image_paths=self.image_paths,
            tabular_data=self.tabular_data,
            labels=self.labels,
        )
        dist = dataset.get_class_distribution()
        self.assertIsInstance(dist, dict)
        self.assertEqual(sum(dist.values()), self.num_samples)

    def test_sample_weights(self):
        """Test sample weight computation for imbalanced data."""
        # Create imbalanced labels
        imbalanced_labels = np.array([0] * 15 + [1] * 5)
        dataset = MedicalMultimodalDataset(
            image_paths=self.image_paths,
            tabular_data=self.tabular_data,
            labels=imbalanced_labels,
        )
        weights = dataset.get_sample_weights()
        self.assertEqual(len(weights), self.num_samples)
        # Class 1 should have higher weight
        self.assertGreater(weights[15], weights[0])

    def test_subset_creation(self):
        """Test dataset subsetting."""
        dataset = MedicalMultimodalDataset(
            image_paths=self.image_paths,
            tabular_data=self.tabular_data,
            labels=self.labels,
        )
        indices = [0, 2, 4, 6, 8]
        subset = dataset.subset(indices)
        self.assertEqual(len(subset), len(indices))


class TestMedicalMultimodalDataset(unittest.TestCase):
    """Test MedicalMultimodalDataset CSV loading."""

    def setUp(self):
        """Create temporary CSV and images."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.image_dir = self.data_dir / "images"
        self.image_dir.mkdir()

        # Create synthetic dataset
        self.num_samples = 30
        data = []
        for i in range(self.num_samples):
            # Create image
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img_name = f"patient_{i:03d}.png"
            img.save(self.image_dir / img_name)

            # Create record
            record = {
                "patient_id": f"P{i:03d}",
                "image_path": img_name,
                "age": np.random.randint(20, 80),
                "marker": np.random.randn(),
                "sex": np.random.choice(["M", "F"]),
                "diagnosis": np.random.randint(0, 2),
            }
            data.append(record)

        self.df = pd.DataFrame(data)
        self.csv_path = self.data_dir / "dataset.csv"
        self.df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_from_csv_loading(self):
        """Test loading dataset from CSV."""
        dataset, scaler = MedicalMultimodalDataset.from_csv(
            csv_path=str(self.csv_path),
            image_dir=str(self.image_dir),
            image_column="image_path",
            target_column="diagnosis",
            numerical_features=["age", "marker"],
            categorical_features=["sex"],
        )
        self.assertEqual(len(dataset), self.num_samples)
        # 2 numerical + 1 categorical (binary encoded sex: M/F -> 0/1)
        self.assertEqual(dataset.get_tabular_dim(), 3)

    def test_missing_image_handling(self):
        """Test handling of missing images."""
        # Remove one image
        (self.image_dir / "patient_000.png").unlink()

        # Dataset loading should skip missing images with warning
        dataset, _ = MedicalMultimodalDataset.from_csv(
            csv_path=str(self.csv_path),
            image_dir=str(self.image_dir),
            image_column="image_path",
            target_column="diagnosis",
            numerical_features=["age"],
        )
        # Should have one less sample due to missing image
        self.assertEqual(len(dataset), self.num_samples - 1)


class TestDatasetSplitting(unittest.TestCase):
    """Test dataset splitting functionality."""

    def setUp(self):
        """Create test dataset."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.temp_dir.name) / "images"
        self.image_dir.mkdir()

        self.num_samples = 100
        image_paths = []
        for i in range(self.num_samples):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img_path = self.image_dir / f"img_{i:03d}.png"
            img.save(img_path)
            image_paths.append(str(img_path))

        tabular_data = np.random.randn(self.num_samples, 3).astype(np.float32)
        labels = np.random.randint(0, 2, self.num_samples)

        self.dataset = MedicalMultimodalDataset(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
        )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_split_ratios(self):
        """Test dataset splitting with specified ratios."""
        train_ds, val_ds, test_ds = split_dataset(
            self.dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42,
        )
        self.assertEqual(len(train_ds), 70)
        self.assertEqual(len(val_ds), 15)
        self.assertEqual(len(test_ds), 15)

    def test_split_reproducibility(self):
        """Test that splitting is reproducible with same seed."""
        train1, val1, test1 = split_dataset(self.dataset, random_seed=42)
        train2, val2, test2 = split_dataset(self.dataset, random_seed=42)
        self.assertEqual(len(train1), len(train2))
        self.assertEqual(len(val1), len(val2))

    def test_invalid_ratios(self):
        """Test that invalid ratios raise error."""
        with self.assertRaises(ValueError):
            split_dataset(
                self.dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum > 1.0
            )


class TestDataLoaders(unittest.TestCase):
    """Test dataloader creation."""

    def setUp(self):
        """Create test datasets."""
        from torchvision import transforms

        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_dir = Path(self.temp_dir.name) / "images"
        self.image_dir.mkdir()

        num_samples = 50
        image_paths = []
        for i in range(num_samples):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img_path = self.image_dir / f"img_{i:03d}.png"
            img.save(img_path)
            image_paths.append(str(img_path))

        tabular_data = np.random.randn(num_samples, 3).astype(np.float32)
        labels = np.random.randint(0, 2, num_samples)

        # Add transform to convert PIL Image to Tensor
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        dataset = MedicalMultimodalDataset(
            image_paths=image_paths,
            tabular_data=tabular_data,
            labels=labels,
            transform=transform,
        )

        self.train_ds, self.val_ds, self.test_ds = split_dataset(
            dataset, random_seed=42
        )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_dataloader_creation(self):
        """Test creating dataloaders."""
        loaders = create_dataloaders(
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            test_dataset=self.test_ds,
            batch_size=8,
            num_workers=0,
        )
        self.assertIn("train", loaders)
        self.assertIn("val", loaders)
        self.assertIn("test", loaders)

    def test_dataloader_iteration(self):
        """Test iterating through dataloader."""
        loaders = create_dataloaders(
            train_dataset=self.train_ds,
            val_dataset=self.val_ds,
            test_dataset=self.test_ds,
            batch_size=8,
            num_workers=0,
        )
        batch = next(iter(loaders["train"]))
        images, tabular, labels = batch
        self.assertEqual(images.shape[0], 8)
        self.assertEqual(tabular.shape[0], 8)
        self.assertEqual(labels.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
