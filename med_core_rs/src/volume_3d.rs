/*!
3D Volume Preprocessing Module

High-performance preprocessing for 3D medical volumes (CT, MRI scans).
Optimized for batch processing with parallel execution.
*/

use ndarray::{Array3, Array4, ArrayView3, s};
use numpy::{PyArray3, PyArray4, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::quickselect::percentile_fast;

/// Normalize 3D volume intensity using Min-Max normalization
///
/// # Arguments
/// * `volume` - Input 3D volume (D, H, W)
///
/// # Returns
/// Normalized volume with values in [0, 1]
#[pyfunction]
pub fn normalize_3d_minmax<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray3<f32>,
) -> Bound<'py, PyArray3<f32>> {
    let array = volume.as_array();
    let result = normalize_3d_minmax_impl(&array);
    PyArray3::from_array_bound(py, &result)
}

/// Normalize 3D volume intensity using Percentile normalization
///
/// # Arguments
/// * `volume` - Input 3D volume (D, H, W)
/// * `p_low` - Lower percentile (default: 1.0)
/// * `p_high` - Upper percentile (default: 99.0)
///
/// # Returns
/// Normalized volume with values in [0, 1]
#[pyfunction]
#[pyo3(signature = (volume, p_low=1.0, p_high=99.0))]
pub fn normalize_3d_percentile<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray3<f32>,
    p_low: f32,
    p_high: f32,
) -> Bound<'py, PyArray3<f32>> {
    let array = volume.as_array();
    let result = normalize_3d_percentile_impl(&array, p_low, p_high);
    PyArray3::from_array_bound(py, &result)
}

/// Batch normalize multiple 3D volumes in parallel
///
/// Processes multiple volumes simultaneously using all available CPU cores.
/// This is significantly faster than processing volumes sequentially.
///
/// # Arguments
/// * `volumes` - Input 4D array (N, D, H, W) where N is batch size
/// * `method` - Normalization method: "minmax" or "percentile"
/// * `p_low` - Lower percentile for percentile method (default: 1.0)
/// * `p_high` - Upper percentile for percentile method (default: 99.0)
///
/// # Returns
/// Normalized volumes (N, D, H, W)
///
/// # Example
/// ```python
/// import numpy as np
/// from med_core_rs import normalize_3d_batch
///
/// # Process 10 CT scans in parallel
/// volumes = np.random.rand(10, 64, 128, 128).astype(np.float32)
/// normalized = normalize_3d_batch(volumes, method="percentile")
/// ```
#[pyfunction]
#[pyo3(signature = (volumes, method="minmax", p_low=1.0, p_high=99.0))]
pub fn normalize_3d_batch<'py>(
    py: Python<'py>,
    volumes: PyReadonlyArray4<f32>,
    method: &str,
    p_low: f32,
    p_high: f32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let array = volumes.as_array();
    let n = array.shape()[0];

    // Process volumes in parallel using Rayon
    let results: Vec<Array3<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let vol = array.slice(s![i, .., .., ..]);
            match method {
                "minmax" => normalize_3d_minmax_impl(&vol),
                "percentile" => normalize_3d_percentile_impl(&vol, p_low, p_high),
                _ => normalize_3d_minmax_impl(&vol), // default
            }
        })
        .collect();

    // Stack results back into 4D array
    let shape = array.shape();
    let mut output = Array4::<f32>::zeros((n, shape[1], shape[2], shape[3]));
    for (i, result) in results.into_iter().enumerate() {
        output.slice_mut(s![i, .., .., ..]).assign(&result);
    }

    Ok(PyArray4::from_array_bound(py, &output))
}

/// Resample 3D volume to target size using trilinear interpolation
///
/// # Arguments
/// * `volume` - Input 3D volume (D, H, W)
/// * `target_d` - Target depth
/// * `target_h` - Target height
/// * `target_w` - Target width
///
/// # Returns
/// Resampled volume (target_d, target_h, target_w)
#[pyfunction]
pub fn resample_3d<'py>(
    py: Python<'py>,
    volume: PyReadonlyArray3<f32>,
    target_d: usize,
    target_h: usize,
    target_w: usize,
) -> Bound<'py, PyArray3<f32>> {
    let array = volume.as_array();
    let result = resample_3d_impl(&array, target_d, target_h, target_w);
    PyArray3::from_array_bound(py, &result)
}

// ============================================================================
// Internal Implementation Functions
// ============================================================================

/// Internal: 3D MinMax normalization
fn normalize_3d_minmax_impl(volume: &ArrayView3<f32>) -> Array3<f32> {
    let min_val = volume.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = volume.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if (max_val - min_val).abs() < 1e-8 {
        return Array3::zeros(volume.raw_dim());
    }

    let range = max_val - min_val;
    volume.mapv(|x| (x - min_val) / range)
}

/// Internal: 3D Percentile normalization
fn normalize_3d_percentile_impl(volume: &ArrayView3<f32>, p_low: f32, p_high: f32) -> Array3<f32> {
    // Convert to 1D for percentile calculation
    let flat: Vec<f32> = volume.iter().cloned().collect();

    // Use quickselect algorithm
    let low = percentile_fast(&flat, p_low);
    let high = percentile_fast(&flat, p_high);

    if (high - low).abs() < 1e-8 {
        return Array3::zeros(volume.raw_dim());
    }

    let range = high - low;
    volume.mapv(|x| ((x - low) / range).max(0.0).min(1.0))
}

/// Internal: 3D trilinear interpolation resampling
fn resample_3d_impl(
    volume: &ArrayView3<f32>,
    target_d: usize,
    target_h: usize,
    target_w: usize,
) -> Array3<f32> {
    let (src_d, src_h, src_w) = volume.dim();

    let scale_d = (src_d - 1) as f32 / (target_d - 1).max(1) as f32;
    let scale_h = (src_h - 1) as f32 / (target_h - 1).max(1) as f32;
    let scale_w = (src_w - 1) as f32 / (target_w - 1).max(1) as f32;

    let output = Array3::<f32>::from_shape_fn((target_d, target_h, target_w), |(d, h, w)| {
        let src_d_f = d as f32 * scale_d;
        let d0 = src_d_f.floor() as usize;
        let d1 = (d0 + 1).min(src_d - 1);
        let wd = src_d_f - d0 as f32;

        let src_h_f = h as f32 * scale_h;
        let h0 = src_h_f.floor() as usize;
        let h1 = (h0 + 1).min(src_h - 1);
        let wh = src_h_f - h0 as f32;

        let src_w_f = w as f32 * scale_w;
        let w0 = src_w_f.floor() as usize;
        let w1 = (w0 + 1).min(src_w - 1);
        let ww = src_w_f - w0 as f32;

        // Trilinear interpolation
        let c000 = volume[[d0, h0, w0]];
        let c001 = volume[[d0, h0, w1]];
        let c010 = volume[[d0, h1, w0]];
        let c011 = volume[[d0, h1, w1]];
        let c100 = volume[[d1, h0, w0]];
        let c101 = volume[[d1, h0, w1]];
        let c110 = volume[[d1, h1, w0]];
        let c111 = volume[[d1, h1, w1]];

        let c00 = c000 * (1.0 - ww) + c001 * ww;
        let c01 = c010 * (1.0 - ww) + c011 * ww;
        let c10 = c100 * (1.0 - ww) + c101 * ww;
        let c11 = c110 * (1.0 - ww) + c111 * ww;

        let c0 = c00 * (1.0 - wh) + c01 * wh;
        let c1 = c10 * (1.0 - wh) + c11 * wh;

        c0 * (1.0 - wd) + c1 * wd
    });

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_normalize_3d_minmax() {
        let volume = Array3::from_shape_fn((4, 4, 4), |(d, h, w)| {
            (d * 16 + h * 4 + w) as f32
        });

        let result = normalize_3d_minmax_impl(&volume.view());

        assert!((result[[0, 0, 0]] - 0.0).abs() < 1e-6);
        assert!((result[[3, 3, 3]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_resample_3d() {
        let volume = Array3::from_shape_fn((8, 8, 8), |(d, h, w)| {
            (d + h + w) as f32
        });

        let resampled = resample_3d_impl(&volume.view(), 4, 4, 4);

        assert_eq!(resampled.dim(), (4, 4, 4));
        // Values should be interpolated
        assert!(resampled[[0, 0, 0]] >= 0.0);
        assert!(resampled[[3, 3, 3]] > 0.0);
    }
}
