"""
Pytest configuration and shared fixtures for med-framework tests.

Provides common test fixtures and utilities used across all test modules.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from torchvision import transforms


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def image_size():
    """Standard image size for tests."""
    return (224, 224)


@pytest.fixture
def default_transform():
    """Default transform that converts PIL images to tensors."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


@pytest.fixture
def num_classes():
    """Standard number of classes for tests."""
    return 2


@pytest.fixture
def sample_images(batch_size, image_size):
    """Generate sample image tensors."""
    return torch.randn(batch_size, 3, *image_size)


@pytest.fixture
def sample_tabular(batch_size):
    """Generate sample tabular data."""
    return torch.randn(batch_size, 10)


@pytest.fixture
def sample_labels(batch_size, num_classes):
    """Generate sample labels."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def sample_multiview_images(batch_size, image_size):
    """Generate sample multi-view images as dictionary."""
    return {
        "axial": torch.randn(batch_size, 3, *image_size),
        "coronal": torch.randn(batch_size, 3, *image_size),
        "sagittal": torch.randn(batch_size, 3, *image_size),
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_csv_data(temp_dir, default_transform):
    """Create sample CSV data for dataset testing."""
    import pandas as pd
    from PIL import Image

    # Create sample images
    image_dir = temp_dir / "images"
    image_dir.mkdir()

    data = []
    for i in range(20):
        # Create dummy image
        img = Image.new("RGB", (224, 224), color=(i * 10, i * 10, i * 10))
        img_path = image_dir / f"patient_{i}.jpg"
        img.save(img_path)

        data.append(
            {
                "patient_id": f"P{i:03d}",
                "image_path": f"patient_{i}.jpg",
                "age": 50 + i,
                "bmi": 25.0 + i * 0.5,
                "gender": i % 2,
                "label": i % 2,
            }
        )

    df = pd.DataFrame(data)
    csv_path = temp_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    return {
        "csv_path": csv_path,
        "image_dir": image_dir,
        "num_samples": len(data),
        "transform": default_transform,  # Add default transform
    }


@pytest.fixture
def sample_multiview_csv_data(temp_dir, default_transform):
    """Create sample multi-view CSV data for dataset testing."""
    import pandas as pd
    from PIL import Image

    # Create sample images
    image_dir = temp_dir / "images"
    image_dir.mkdir()

    data = []
    for i in range(20):
        # Create dummy images for each view
        views = {}
        for view_name in ["axial", "coronal", "sagittal"]:
            img = Image.new("RGB", (224, 224), color=(i * 10, i * 10, i * 10))
            img_path = image_dir / f"patient_{i}_{view_name}.jpg"
            img.save(img_path)
            views[f"{view_name}_path"] = f"patient_{i}_{view_name}.jpg"

        data.append(
            {
                "patient_id": f"P{i:03d}",
                **views,
                "age": 50 + i,
                "bmi": 25.0 + i * 0.5,
                "gender": i % 2,
                "label": i % 2,
            }
        )

    df = pd.DataFrame(data)
    csv_path = temp_dir / "multiview_data.csv"
    df.to_csv(csv_path, index=False)

    return {
        "csv_path": csv_path,
        "image_dir": image_dir,
        "num_samples": len(data),
        "view_columns": {
            "axial": "axial_path",
            "coronal": "coronal_path",
            "sagittal": "sagittal_path",
        },
        "transform": default_transform,  # Add default transform
    }


@pytest.fixture
def vision_backbone_names():
    """List of vision backbone names to test."""
    return ["resnet18", "mobilenetv2"]


@pytest.fixture
def fusion_types():
    """List of fusion types to test."""
    return ["concatenate", "gated", "attention", "cross_attention", "bilinear"]


@pytest.fixture
def aggregator_types():
    """List of view aggregator types to test."""
    return ["max", "mean", "attention", "cross_attention", "learned_weight"]


# Utility functions for tests


def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {tensor.shape}"
    )


def assert_tensor_range(tensor, min_val=None, max_val=None):
    """Assert tensor values are within range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"Tensor min {tensor.min()} < {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"Tensor max {tensor.max()} > {max_val}"


def assert_model_trainable(model, should_be_trainable=True):
    """Assert model has trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if should_be_trainable:
        assert trainable_params > 0, "Model has no trainable parameters"
    else:
        assert trainable_params == 0, "Model has trainable parameters when it shouldn't"


# Export utility functions
pytest.assert_tensor_shape = assert_tensor_shape
pytest.assert_tensor_range = assert_tensor_range
pytest.assert_model_trainable = assert_model_trainable
