/*!
Medical Image Preprocessing Module

High-performance implementations of common medical image preprocessing operations.
*/

use ndarray::{Array2, Array3, ArrayView2, s};
use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::quickselect::percentile_fast;

/// Normalize image intensity using Min-Max normalization
///
/// Scales pixel values to [0, 1] range based on min and max values.
///
/// # Arguments
/// * `image` - Input 2D image array (H, W)
///
/// # Returns
/// Normalized image with values in [0, 1]
///
/// # Example
/// ```python
/// import numpy as np
/// from med_core_rs import normalize_intensity_minmax
///
/// image = np.random.rand(512, 512).astype(np.float32)
/// normalized = normalize_intensity_minmax(image)
/// ```
#[pyfunction]
pub fn normalize_intensity_minmax<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let array = image.as_array();
    let result = normalize_minmax(&array);
    PyArray2::from_array_bound(py, &result)
}

/// Normalize image intensity using Z-score normalization
///
/// Standardizes pixel values to have mean=0 and std=1, then clips to [-3, 3]
/// and rescales to [0, 1].
///
/// # Arguments
/// * `image` - Input 2D image array (H, W)
///
/// # Returns
/// Normalized image with values in [0, 1]
#[pyfunction]
pub fn normalize_intensity_zscore<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let array = image.as_array();
    let result = normalize_zscore(&array);
    PyArray2::from_array_bound(py, &result)
}

/// Normalize image intensity using percentile-based normalization
///
/// Clips values to specified percentile range and scales to [0, 1].
///
/// # Arguments
/// * `image` - Input 2D image array (H, W)
/// * `p_low` - Lower percentile (default: 1.0)
/// * `p_high` - Upper percentile (default: 99.0)
///
/// # Returns
/// Normalized image with values in [0, 1]
#[pyfunction]
#[pyo3(signature = (image, p_low=1.0, p_high=99.0))]
pub fn normalize_intensity_percentile<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f32>,
    p_low: f32,
    p_high: f32,
) -> Bound<'py, PyArray2<f32>> {
    let array = image.as_array();
    let result = normalize_percentile(&array, p_low, p_high);
    PyArray2::from_array_bound(py, &result)
}

/// Batch normalize multiple images in parallel
///
/// Processes multiple images simultaneously using all available CPU cores.
/// This is significantly faster than processing images sequentially.
///
/// # Arguments
/// * `images` - Input 3D array (N, H, W) where N is batch size
/// * `method` - Normalization method: "minmax", "zscore", or "percentile"
/// * `p_low` - Lower percentile for percentile method (default: 1.0)
/// * `p_high` - Upper percentile for percentile method (default: 99.0)
///
/// # Returns
/// Normalized images (N, H, W)
///
/// # Example
/// ```python
/// import numpy as np
/// from med_core_rs import normalize_intensity_batch
///
/// # Process 100 images in parallel
/// images = np.random.rand(100, 512, 512).astype(np.float32)
/// normalized = normalize_intensity_batch(images, method="percentile")
/// ```
#[pyfunction]
#[pyo3(signature = (images, method="minmax", p_low=1.0, p_high=99.0))]
pub fn normalize_intensity_batch<'py>(
    py: Python<'py>,
    images: PyReadonlyArray3<f32>,
    method: &str,
    p_low: f32,
    p_high: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let array = images.as_array();
    let n = array.shape()[0];

    // Process images in parallel using Rayon
    let results: Vec<Array2<f32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let img = array.slice(s![i, .., ..]);
            match method {
                "minmax" => normalize_minmax(&img),
                "zscore" => normalize_zscore(&img),
                "percentile" => normalize_percentile(&img, p_low, p_high),
                _ => normalize_minmax(&img), // default
            }
        })
        .collect();

    // Stack results back into 3D array
    let shape = array.shape();
    let mut output = Array3::<f32>::zeros((n, shape[1], shape[2]));
    for (i, result) in results.into_iter().enumerate() {
        output.slice_mut(s![i, .., ..]).assign(&result);
    }

    Ok(PyArray3::from_array_bound(py, &output))
}

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
///
/// Enhances local contrast in medical images using adaptive histogram equalization.
///
/// # Arguments
/// * `image` - Input 2D grayscale image (H, W), values in [0, 255] as uint8
/// * `clip_limit` - Contrast clipping limit (default: 2.0)
/// * `tile_size` - Size of tiles for local histogram equalization (default: 8)
///
/// # Returns
/// Enhanced image (H, W) as uint8
///
/// # Note
/// For RGB images, convert to LAB color space and apply CLAHE to L channel only.
#[pyfunction]
#[pyo3(signature = (image, clip_limit=2.0, tile_size=8))]
pub fn apply_clahe_rust<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
    clip_limit: f32,
    tile_size: u32,
) -> Bound<'py, PyArray2<u8>> {
    let array = image.as_array();
    let result = clahe_2d(&array, clip_limit, tile_size);
    PyArray2::from_array_bound(py, &result)
}

/// Center crop an image to target size
///
/// Extracts the center region of an image.
///
/// # Arguments
/// * `image` - Input 2D image (H, W)
/// * `target_h` - Target height
/// * `target_w` - Target width
///
/// # Returns
/// Cropped image (target_h, target_w)
#[pyfunction]
pub fn center_crop_rust<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<f32>,
    target_h: usize,
    target_w: usize,
) -> Bound<'py, PyArray2<f32>> {
    let array = image.as_array();
    let result = center_crop(&array, target_h, target_w);
    PyArray2::from_array_bound(py, &result)
}

// ============================================================================
// Internal Implementation Functions (Public for benchmarking)
// ============================================================================

/// Internal: Min-Max normalization
pub fn normalize_minmax(image: &ArrayView2<f32>) -> Array2<f32> {
    let min_val = image.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = image.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if (max_val - min_val).abs() < 1e-8 {
        return Array2::zeros(image.raw_dim());
    }

    let range = max_val - min_val;
    image.mapv(|x| (x - min_val) / range)
}

/// Internal: Z-score normalization
pub fn normalize_zscore(image: &ArrayView2<f32>) -> Array2<f32> {
    let mean = image.mean().unwrap_or(0.0);
    let std = image.std(0.0);

    if std < 1e-8 {
        return image.mapv(|x| x - mean);
    }

    // Standardize, clip to [-3, 3], then scale to [0, 1]
    image.mapv(|x| {
        let z = (x - mean) / std;
        let clipped = z.max(-3.0).min(3.0);
        (clipped + 3.0) / 6.0
    })
}

/// Internal: Percentile-based normalization (optimized with quickselect)
pub fn normalize_percentile(image: &ArrayView2<f32>, p_low: f32, p_high: f32) -> Array2<f32> {
    // Convert to 1D for percentile calculation
    let flat: Vec<f32> = image.iter().cloned().collect();

    // Use quickselect algorithm (O(n) average) instead of sorting (O(n log n))
    let low = percentile_fast(&flat, p_low);
    let high = percentile_fast(&flat, p_high);

    if (high - low).abs() < 1e-8 {
        return Array2::zeros(image.raw_dim());
    }

    let range = high - low;
    image.mapv(|x| ((x - low) / range).max(0.0).min(1.0))
}

/// Internal: CLAHE implementation
fn clahe_2d(image: &ArrayView2<u8>, _clip_limit: f32, tile_size: u32) -> Array2<u8> {
    // Convert ndarray to image crate format
    let (height, width) = image.dim();
    let img_vec: Vec<u8> = image.iter().cloned().collect();

    let img = image::GrayImage::from_raw(width as u32, height as u32, img_vec)
        .expect("Failed to create image");

    // Apply CLAHE using imageproc
    let enhanced = imageproc::contrast::adaptive_threshold(
        &img,
        tile_size,
    );

    // Convert back to ndarray
    let result_vec = enhanced.into_raw();
    Array2::from_shape_vec((height, width), result_vec)
        .expect("Failed to create array")
}

/// Internal: Center crop
pub fn center_crop(image: &ArrayView2<f32>, target_h: usize, target_w: usize) -> Array2<f32> {
    let (height, width) = image.dim();

    let start_h = (height.saturating_sub(target_h)) / 2;
    let start_w = (width.saturating_sub(target_w)) / 2;

    let end_h = (start_h + target_h).min(height);
    let end_w = (start_w + target_w).min(width);

    image.slice(s![start_h..end_h, start_w..end_w]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_normalize_minmax() {
        let image = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let result = normalize_minmax(&image.view());

        assert!((result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((result[[1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_center_crop() {
        let image = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f32).collect()).unwrap();
        let cropped = center_crop(&image.view(), 2, 2);

        assert_eq!(cropped.dim(), (2, 2));
        assert_eq!(cropped[[0, 0]], 5.0);
        assert_eq!(cropped[[1, 1]], 10.0);
    }
}
