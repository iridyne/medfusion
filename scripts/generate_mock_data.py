#!/usr/bin/env python3
"""
Generate mock medical data for testing Med-Core framework.

This script creates:
- 10 random noise images (224x224 RGB)
- metadata.csv with patient information and labels

Usage:
    uv run python scripts/generate_mock_data.py
    uv run python scripts/generate_mock_data.py --output-dir data/custom_mock
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def generate_mock_images(output_dir: Path, num_images: int = 10, image_size: int = 224):
    """
    Generate random noise images as mock medical images.

    Args:
        output_dir: Directory to save images
        num_images: Number of images to generate
        image_size: Size of square images (width = height)
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i in range(num_images):
        # Generate random RGB noise
        noise = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)

        # Convert to PIL Image
        img = Image.fromarray(noise)

        # Save image
        filename = f"patient_{i:03d}.png"
        filepath = images_dir / filename
        img.save(filepath)

        # Store relative path for CSV
        image_paths.append(f"images/{filename}")

    return image_paths


def generate_mock_metadata(output_dir: Path, image_paths: list[str], num_samples: int = 10):
    """
    Generate mock patient metadata CSV.

    Args:
        output_dir: Directory to save CSV
        image_paths: List of image paths
        num_samples: Number of samples to generate
    """
    np.random.seed(42)

    data = {
        "patient_id": [f"P{i:03d}" for i in range(num_samples)],
        "image_path": image_paths,
        "age": np.random.randint(20, 80, num_samples),
        "gender": np.random.choice([0, 1], num_samples),
        "diagnosis": np.random.choice([0, 1], num_samples),
    }

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate mock medical data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mock",
        help="Output directory for mock data",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Size of square images",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_samples} mock samples...")
    print(f"Output directory: {output_dir.absolute()}")

    # Generate images
    print(f"\n1. Generating {args.num_samples} images ({args.image_size}x{args.image_size})...")
    image_paths = generate_mock_images(
        output_dir, num_images=args.num_samples, image_size=args.image_size
    )
    print(f"   ✓ Images saved to {output_dir / 'images'}")

    # Generate metadata
    print("\n2. Generating metadata CSV...")
    csv_path = generate_mock_metadata(output_dir, image_paths, num_samples=args.num_samples)
    print(f"   ✓ Metadata saved to {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Mock data generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - {args.num_samples} images in {output_dir / 'images'}")
    print("  - metadata.csv with columns: patient_id, image_path, age, gender, diagnosis")
    print("\nTo use this data:")
    print("  1. Update your config YAML:")
    print("     data:")
    print(f"       csv_path: \"{csv_path}\"")
    print(f"       image_dir: \"{output_dir}\"")
    print("  2. Run training:")
    print("     uv run med-train --config configs/your_config.yaml")


if __name__ == "__main__":
    main()
