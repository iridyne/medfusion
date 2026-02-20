/*!
# MedCore Rust Accelerated Module

High-performance Rust implementations of performance-critical medical image
preprocessing operations.

## Features

- **Intensity Normalization**: MinMax, Z-score, Percentile methods with SIMD acceleration
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Batch Processing**: Parallel processing of image batches using Rayon
- **Zero-copy Integration**: Efficient memory sharing with NumPy/PyTorch

## Performance

Typical speedups over pure Python/NumPy implementations:
- Batch normalization: 5-10x faster
- CLAHE: 3-5x faster
- Overall preprocessing pipeline: 4-8x faster
*/

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod preprocessing;
mod quickselect;
pub mod volume_3d;
pub mod mil;

use preprocessing::{
    normalize_intensity_minmax, normalize_intensity_percentile, normalize_intensity_zscore,
    normalize_intensity_batch, apply_clahe_rust, center_crop_rust,
};

use volume_3d::{
    normalize_3d_minmax, normalize_3d_percentile, normalize_3d_batch, resample_3d,
};

use mil::{
    attention_mil, max_pooling_mil, mean_pooling_mil, batch_mil_aggregation,
};

/// Python module initialization
#[pymodule]
fn med_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 2D image preprocessing
    m.add_function(wrap_pyfunction!(normalize_intensity_minmax, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_intensity_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_intensity_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_intensity_batch, m)?)?;
    m.add_function(wrap_pyfunction!(apply_clahe_rust, m)?)?;
    m.add_function(wrap_pyfunction!(center_crop_rust, m)?)?;

    // 3D volume preprocessing
    m.add_function(wrap_pyfunction!(normalize_3d_minmax, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_3d_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_3d_batch, m)?)?;
    m.add_function(wrap_pyfunction!(resample_3d, m)?)?;

    // MIL aggregators
    m.add_function(wrap_pyfunction!(attention_mil, m)?)?;
    m.add_function(wrap_pyfunction!(max_pooling_mil, m)?)?;
    m.add_function(wrap_pyfunction!(mean_pooling_mil, m)?)?;
    m.add_function(wrap_pyfunction!(batch_mil_aggregation, m)?)?;

    Ok(())
}
