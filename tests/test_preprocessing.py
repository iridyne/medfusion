"""
Tests for preprocessing utilities.
"""

import numpy as np
import pytest
from PIL import Image

from med_core.preprocessing import ImagePreprocessor


class TestImagePreprocessor:
    """Test ImagePreprocessor class."""

    def test_preprocessor_initialization(self):
        """Test preprocessor can be initialized with default parameters."""
        preprocessor = ImagePreprocessor()
        assert preprocessor is not None

    def test_preprocessor_with_custom_params(self):
        """Test preprocessor with custom parameters."""
        preprocessor = ImagePreprocessor(
            normalize_method="minmax",
            remove_watermark=True,
            apply_clahe=True,
            output_size=(256, 256),
        )
        assert preprocessor is not None

    @pytest.mark.parametrize(
        "normalize_method",
        ["minmax", "zscore", "percentile", "none"],
    )
    def test_preprocessor_normalization_methods(self, normalize_method: str):
        """Test different normalization methods."""
        preprocessor = ImagePreprocessor(normalize_method=normalize_method)
        assert preprocessor is not None

    def test_preprocessor_invalid_normalize_method(self):
        """Test that invalid normalization method raises error."""
        with pytest.raises((ValueError, KeyError)):
            ImagePreprocessor(normalize_method="invalid_method")

    def test_preprocess_single_image(self, tmp_path):
        """Test preprocessing a single image."""
        # Create a dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        # Preprocess
        preprocessor = ImagePreprocessor(output_size=(224, 224))
        processed = preprocessor.preprocess(str(img_path))

        assert processed is not None
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (224, 224, 3)

    def test_preprocess_output_size(self, tmp_path):
        """Test that output size is correctly applied."""
        # Create a dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        # Test different output sizes
        sizes = [(128, 128), (256, 256), (512, 512)]
        for size in sizes:
            preprocessor = ImagePreprocessor(output_size=size)
            processed = preprocessor.preprocess(str(img_path))
            assert processed.shape[:2] == size

    def test_preprocess_grayscale_image(self, tmp_path):
        """Test preprocessing grayscale image."""
        # Create a grayscale image
        img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")
        img_path = tmp_path / "test_gray.jpg"
        img.save(img_path)

        preprocessor = ImagePreprocessor(output_size=(224, 224))
        processed = preprocessor.preprocess(str(img_path))

        assert processed is not None
        assert isinstance(processed, np.ndarray)

    def test_preprocess_with_clahe(self, tmp_path):
        """Test preprocessing with CLAHE enhancement."""
        # Create a dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        preprocessor = ImagePreprocessor(
            apply_clahe=True,
            output_size=(224, 224),
        )
        processed = preprocessor.preprocess(str(img_path))

        assert processed is not None
        assert processed.shape == (224, 224, 3)

    def test_batch_processing(self, tmp_path):
        """Test batch processing of multiple images."""
        # Create multiple dummy images
        image_paths = []
        for i in range(3):
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = tmp_path / f"test_image_{i}.jpg"
            img.save(img_path)
            image_paths.append(img_path)

        # Process batch
        output_dir = tmp_path / "processed"
        preprocessor = ImagePreprocessor(output_size=(224, 224))
        preprocessor.process_batch(image_paths, str(output_dir))

        # Check output files exist
        assert output_dir.exists()
        output_files = list(output_dir.glob("*.jpg"))
        assert len(output_files) == 3


class TestPreprocessingPipeline:
    """Test preprocessing pipeline integration."""

    def test_pipeline_consistency(self, tmp_path):
        """Test that preprocessing is consistent across multiple runs."""
        # Create a dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        preprocessor = ImagePreprocessor(
            normalize_method="minmax",
            output_size=(224, 224),
        )

        # Process same image twice
        result1 = preprocessor.preprocess(str(img_path))
        result2 = preprocessor.preprocess(str(img_path))

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)

    def test_pipeline_with_all_options(self, tmp_path):
        """Test preprocessing with all options enabled."""
        # Create a dummy image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)

        preprocessor = ImagePreprocessor(
            normalize_method="percentile",
            remove_watermark=True,
            apply_clahe=True,
            output_size=(224, 224),
        )

        processed = preprocessor.preprocess(str(img_path))

        assert processed is not None
        assert processed.shape == (224, 224, 3)
