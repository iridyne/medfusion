"""
Example: Using Rust-accelerated preprocessing in MedFusion pipeline.

This demonstrates how to seamlessly integrate Rust acceleration into
existing Python workflows.
"""

import numpy as np
import time
from pathlib import Path

try:
    from med_core_rs import (
        normalize_intensity_minmax,
        normalize_intensity_percentile,
        normalize_intensity_batch,
        center_crop_rust,
    )
    RUST_AVAILABLE = True
    print("✅ Rust acceleration available")
except ImportError:
    print("⚠️  Rust module not available. Using Python fallback.")
    RUST_AVAILABLE = False
    from med_core.shared.data_utils.image_preprocessing import (
        normalize_intensity,
        center_crop,
    )


class RustAcceleratedPreprocessor:
    """
    Drop-in replacement for ImagePreprocessor with Rust acceleration.

    Example:
        >>> preprocessor = RustAcceleratedPreprocessor(
        ...     normalize_method="percentile",
        ...     output_size=(224, 224)
        ... )
        >>> processed = preprocessor.process_batch(image_paths)
    """

    def __init__(
        self,
        normalize_method: str = "percentile",
        output_size: tuple[int, int] = (224, 224),
        use_rust: bool = True,
    ):
        self.normalize_method = normalize_method
        self.output_size = output_size
        self.use_rust = use_rust and RUST_AVAILABLE

        if self.use_rust:
            print("🚀 Using Rust-accelerated preprocessing")
        else:
            print("🐍 Using Python preprocessing")

    def preprocess_single(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image."""
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize
        if self.use_rust:
            if self.normalize_method == "minmax":
                image = normalize_intensity_minmax(image)
            elif self.normalize_method == "percentile":
                image = normalize_intensity_percentile(image, 1.0, 99.0)
        else:
            image = normalize_intensity(image, self.normalize_method)

        # Crop to target size
        if self.output_size:
            if self.use_rust:
                image = center_crop_rust(image, self.output_size[0], self.output_size[1])
            else:
                # Python fallback
                h, w = image.shape
                th, tw = self.output_size
                start_h = (h - th) // 2
                start_w = (w - tw) // 2
                image = image[start_h:start_h+th, start_w:start_w+tw]

        return image

    def preprocess_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocess a batch of images in parallel (Rust only).

        Args:
            images: Array of shape (N, H, W)

        Returns:
            Preprocessed images (N, H', W')
        """
        if not self.use_rust:
            # Fallback to sequential processing
            return np.array([self.preprocess_single(img) for img in images])

        # Ensure float32
        if images.dtype != np.float32:
            images = images.astype(np.float32)

        # Parallel normalization
        images = normalize_intensity_batch(
            images,
            method=self.normalize_method,
            p_low=1.0,
            p_high=99.0
        )

        # Crop each image (TODO: batch crop in Rust)
        if self.output_size:
            processed = []
            for img in images:
                cropped = center_crop_rust(img, self.output_size[0], self.output_size[1])
                processed.append(cropped)
            images = np.array(processed)

        return images


def demo_single_image():
    """Demo: Process a single image."""
    print("\n" + "="*70)
    print("DEMO 1: Single Image Processing")
    print("="*70)

    # Generate test image
    image = np.random.rand(1024, 1024).astype(np.float32) * 255
    print(f"Input image shape: {image.shape}")

    preprocessor = RustAcceleratedPreprocessor(
        normalize_method="percentile",
        output_size=(224, 224)
    )

    start = time.perf_counter()
    processed = preprocessor.preprocess_single(image)
    elapsed = time.perf_counter() - start

    print(f"Output image shape: {processed.shape}")
    print(f"Processing time: {elapsed*1000:.2f} ms")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")


def demo_batch_processing():
    """Demo: Process a batch of images."""
    print("\n" + "="*70)
    print("DEMO 2: Batch Processing")
    print("="*70)

    batch_size = 100
    img_size = 512

    # Generate test batch
    images = np.random.rand(batch_size, img_size, img_size).astype(np.float32) * 255
    print(f"Input batch shape: {images.shape}")

    preprocessor = RustAcceleratedPreprocessor(
        normalize_method="percentile",
        output_size=(224, 224)
    )

    start = time.perf_counter()
    processed = preprocessor.preprocess_batch(images)
    elapsed = time.perf_counter() - start

    print(f"Output batch shape: {processed.shape}")
    print(f"Total processing time: {elapsed*1000:.2f} ms")
    print(f"Time per image: {elapsed/batch_size*1000:.2f} ms")
    print(f"Throughput: {batch_size/elapsed:.1f} images/sec")


def demo_integration_with_dataloader():
    """Demo: Integration with PyTorch DataLoader."""
    print("\n" + "="*70)
    print("DEMO 3: Integration with DataLoader")
    print("="*70)

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("⚠️  PyTorch not available, skipping demo")
        return

    class MedicalDataset(Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
            self.preprocessor = RustAcceleratedPreprocessor(
                normalize_method="percentile",
                output_size=(224, 224)
            )

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Simulate loading an image
            image = np.random.rand(512, 512).astype(np.float32) * 255

            # Preprocess with Rust acceleration
            processed = self.preprocessor.preprocess_single(image)

            # Convert to tensor
            tensor = torch.from_numpy(processed).unsqueeze(0)  # Add channel dim

            # Dummy label
            label = idx % 4

            return tensor, label

    # Create dataset and dataloader
    dataset = MedicalDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4)

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 16")
    print(f"Num workers: 4")

    # Benchmark data loading
    start = time.perf_counter()
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"\nFirst batch shape: {images.shape}")
        pass  # Simulate training
    elapsed = time.perf_counter() - start

    print(f"\nTotal time: {elapsed:.2f} sec")
    print(f"Throughput: {len(dataset)/elapsed:.1f} samples/sec")


def main():
    print("\n" + "="*70)
    print("MedCore Rust Acceleration - Integration Examples")
    print("="*70)

    demo_single_image()
    demo_batch_processing()
    demo_integration_with_dataloader()

    print("\n" + "="*70)
    print("✅ All demos completed!")
    print("="*70)
    print("\nNext steps:")
    print("1. Integrate into your training pipeline")
    print("2. Run benchmarks: python benchmark_comparison.py")
    print("3. Profile with: python -m cProfile your_script.py")


if __name__ == "__main__":
    main()
