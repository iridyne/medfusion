"""
Python benchmark script to compare Rust vs NumPy performance.

Usage:
    python benchmark_comparison.py
"""

import time
from collections.abc import Callable

import numpy as np

# Import both implementations
try:
    from med_core_rs import (
        normalize_intensity_batch as rust_batch,
    )
    from med_core_rs import (
        normalize_intensity_minmax as rust_minmax,
    )
    from med_core_rs import (
        normalize_intensity_percentile as rust_percentile,
    )
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: Rust module not available. Install with: maturin develop --release")
    RUST_AVAILABLE = False

from med_core.shared.data_utils.image_preprocessing import (
    normalize_intensity as python_normalize,
)


def benchmark_function(func: Callable, *args, iterations: int = 100) -> float:
    """Benchmark a function and return average execution time."""
    times = []

    # Warmup
    for _ in range(5):
        func(*args)

    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)


def compare_normalization():
    """Compare normalization performance."""
    print("\n" + "="*70)
    print("NORMALIZATION BENCHMARK")
    print("="*70)

    sizes = [256, 512, 1024]

    for size in sizes:
        print(f"\n📊 Image size: {size}x{size}")
        image = np.random.rand(size, size).astype(np.float32) * 255

        # Python MinMax
        py_mean, py_std = benchmark_function(
            python_normalize, image, "minmax", iterations=50
        )
        print(f"  Python MinMax:     {py_mean*1000:.2f} ± {py_std*1000:.2f} ms")

        if RUST_AVAILABLE:
            # Rust MinMax
            rust_mean, rust_std = benchmark_function(
                rust_minmax, image, iterations=50
            )
            print(f"  Rust MinMax:       {rust_mean*1000:.2f} ± {rust_std*1000:.2f} ms")
            speedup = py_mean / rust_mean
            print(f"  🚀 Speedup:        {speedup:.2f}x")

        print()

        # Python Percentile
        py_mean, py_std = benchmark_function(
            python_normalize, image, "percentile", iterations=50
        )
        print(f"  Python Percentile: {py_mean*1000:.2f} ± {py_std*1000:.2f} ms")

        if RUST_AVAILABLE:
            # Rust Percentile
            rust_mean, rust_std = benchmark_function(
                rust_percentile, image, 1.0, 99.0, iterations=50
            )
            print(f"  Rust Percentile:   {rust_mean*1000:.2f} ± {rust_std*1000:.2f} ms")
            speedup = py_mean / rust_mean
            print(f"  🚀 Speedup:        {speedup:.2f}x")


def compare_batch_processing():
    """Compare batch processing performance."""
    print("\n" + "="*70)
    print("BATCH PROCESSING BENCHMARK")
    print("="*70)

    batch_sizes = [10, 50, 100]
    img_size = 512

    for batch_size in batch_sizes:
        print(f"\n📊 Batch size: {batch_size} images ({img_size}x{img_size})")
        images = np.random.rand(batch_size, img_size, img_size).astype(np.float32) * 255

        # Python sequential processing
        def python_batch_process(imgs):
            return np.array([
                python_normalize(img, "percentile")
                for img in imgs
            ])

        py_mean, py_std = benchmark_function(
            python_batch_process, images, iterations=20
        )
        print(f"  Python (sequential): {py_mean*1000:.2f} ± {py_std*1000:.2f} ms")

        if RUST_AVAILABLE:
            # Rust parallel processing
            rust_mean, rust_std = benchmark_function(
                rust_batch, images, "percentile", 1.0, 99.0, iterations=20
            )
            print(f"  Rust (parallel):     {rust_mean*1000:.2f} ± {rust_std*1000:.2f} ms")
            speedup = py_mean / rust_mean
            print(f"  🚀 Speedup:          {speedup:.2f}x")

            throughput = batch_size / rust_mean
            print(f"  📈 Throughput:       {throughput:.1f} images/sec")


def verify_correctness():
    """Verify that Rust and Python implementations produce similar results."""
    print("\n" + "="*70)
    print("CORRECTNESS VERIFICATION")
    print("="*70)

    if not RUST_AVAILABLE:
        print("⚠️  Rust module not available, skipping verification")
        return

    image = np.random.rand(256, 256).astype(np.float32) * 255

    # MinMax
    py_result = python_normalize(image, "minmax")
    rust_result = rust_minmax(image)
    diff = np.abs(py_result - rust_result).max()
    print(f"\n✓ MinMax max difference: {diff:.6f}")
    assert diff < 1e-5, "MinMax results differ too much!"

    # Percentile
    py_result = python_normalize(image, "percentile")
    rust_result = rust_percentile(image, 1.0, 99.0)
    diff = np.abs(py_result - rust_result).max()
    print(f"✓ Percentile max difference: {diff:.6f}")
    assert diff < 1e-3, "Percentile results differ too much!"

    print("\n✅ All correctness checks passed!")


def main():
    print("\n" + "="*70)
    print("MedCore Rust Acceleration Benchmark")
    print("="*70)

    if not RUST_AVAILABLE:
        print("\n⚠️  Rust module not found!")
        print("Build and install with:")
        print("  cd med_core_rs")
        print("  maturin develop --release")
        print("\nRunning Python-only benchmarks...\n")

    verify_correctness()
    compare_normalization()
    compare_batch_processing()

    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
