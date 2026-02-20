# Quick Start Guide

## Installation (5 minutes)

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Install maturin
pip install maturin

# 3. Build and install the module
cd med_core_rs
maturin develop --release
```

## Verify Installation

```python
python -c "from med_core_rs import normalize_intensity_minmax; print('✅ Success!')"
```

## Run Benchmarks

```bash
# Compare Python vs Rust performance
python benchmark_comparison.py

# Expected output:
# ======================================================================
# NORMALIZATION BENCHMARK
# ======================================================================
#
# 📊 Image size: 512x512
#   Python MinMax:     2.45 ± 0.12 ms
#   Rust MinMax:       0.38 ± 0.02 ms
#   🚀 Speedup:        6.4x
```

## Try the Examples

```bash
python example_integration.py
```

## Integration into Your Code

### Before (Pure Python)
```python
from med_core.shared.data_utils.image_preprocessing import normalize_intensity

for image in images:
    normalized = normalize_intensity(image, "percentile")
```

### After (Rust Accelerated)
```python
from med_core_rs import normalize_intensity_batch

# Process all images in parallel!
normalized = normalize_intensity_batch(images, method="percentile")
```

That's it! You should see 5-10x speedup immediately.

## Next Steps

1. Read the full [README.md](README.md) for API details
2. Run `cargo bench` for detailed performance metrics
3. Integrate into your training pipeline
4. Profile with `python -m cProfile` to verify improvements
