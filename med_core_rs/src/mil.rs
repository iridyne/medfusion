/*!
Multiple Instance Learning (MIL) Aggregators

High-performance implementations of MIL aggregation functions for bag-level predictions.
Optimized for batch processing with parallel execution.
*/

use ndarray::{Array1, Array2, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Attention-based MIL aggregation
///
/// Computes weighted average of instances using attention mechanism.
///
/// # Arguments
/// * `instances` - Instance features (N, D) where N is number of instances, D is feature dimension
/// * `attention_weights` - Attention weights (N,) for each instance
///
/// # Returns
/// Aggregated bag representation (D,)
#[pyfunction]
pub fn attention_mil<'py>(
    py: Python<'py>,
    instances: PyReadonlyArray2<f32>,
    attention_weights: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<f32>> {
    let inst_array = instances.as_array();
    let attn_array = attention_weights.as_array();

    let result = attention_mil_impl(&inst_array, &attn_array);
    PyArray1::from_array_bound(py, &result)
}

/// Max pooling MIL aggregation
///
/// Takes the maximum value across all instances for each feature dimension.
///
/// # Arguments
/// * `instances` - Instance features (N, D)
///
/// # Returns
/// Aggregated bag representation (D,)
#[pyfunction]
pub fn max_pooling_mil<'py>(
    py: Python<'py>,
    instances: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<f32>> {
    let array = instances.as_array();
    let result = max_pooling_mil_impl(&array);
    PyArray1::from_array_bound(py, &result)
}

/// Mean pooling MIL aggregation
///
/// Computes the average across all instances for each feature dimension.
///
/// # Arguments
/// * `instances` - Instance features (N, D)
///
/// # Returns
/// Aggregated bag representation (D,)
#[pyfunction]
pub fn mean_pooling_mil<'py>(
    py: Python<'py>,
    instances: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray1<f32>> {
    let array = instances.as_array();
    let result = mean_pooling_mil_impl(&array);
    PyArray1::from_array_bound(py, &result)
}

/// Batch MIL aggregation for multiple bags
///
/// Processes multiple bags in parallel using specified aggregation method.
///
/// # Arguments
/// * `bags` - List of instance arrays, each (N_i, D)
/// * `method` - Aggregation method: "max", "mean", or "attention"
/// * `attention_weights` - Optional attention weights for each bag (only for attention method)
///
/// # Returns
/// Aggregated representations (B, D) where B is number of bags
///
/// # Example
/// ```python
/// import numpy as np
/// from med_core_rs import batch_mil_aggregation
///
/// # Process 100 bags in parallel
/// bags = [np.random.rand(np.random.randint(10, 100), 512).astype(np.float32)
///         for _ in range(100)]
/// aggregated = batch_mil_aggregation(bags, method="max")
/// ```
#[pyfunction]
#[pyo3(signature = (bags, method="max", attention_weights=None))]
pub fn batch_mil_aggregation<'py>(
    py: Python<'py>,
    bags: Vec<PyReadonlyArray2<f32>>,
    method: &str,
    attention_weights: Option<Vec<PyReadonlyArray2<f32>>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n_bags = bags.len();
    if n_bags == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Empty bags list"));
    }

    let feature_dim = bags[0].as_array().shape()[1];

    // Convert to owned arrays for parallel processing
    let bag_arrays: Vec<_> = bags.iter().map(|b| b.as_array().to_owned()).collect();

    // Process bags in parallel
    let results: Vec<Array1<f32>> = match method {
        "max" => bag_arrays
            .par_iter()
            .map(|bag| max_pooling_mil_impl(&bag.view()))
            .collect(),
        "mean" => bag_arrays
            .par_iter()
            .map(|bag| mean_pooling_mil_impl(&bag.view()))
            .collect(),
        "attention" => {
            if let Some(weights) = attention_weights {
                if weights.len() != n_bags {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Number of attention weights must match number of bags",
                    ));
                }
                let weight_arrays: Vec<_> = weights.iter().map(|w| w.as_array().to_owned()).collect();
                bag_arrays
                    .par_iter()
                    .zip(weight_arrays.par_iter())
                    .map(|(bag, weight)| {
                        attention_mil_impl(&bag.view(), &weight.view())
                    })
                    .collect()
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Attention weights required for attention method",
                ));
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown method: {}",
                method
            )))
        }
    };

    // Stack results into 2D array
    let mut output = Array2::<f32>::zeros((n_bags, feature_dim));
    for (i, result) in results.into_iter().enumerate() {
        output.row_mut(i).assign(&result);
    }

    Ok(PyArray2::from_array_bound(py, &output))
}

// ============================================================================
// Internal Implementation Functions
// ============================================================================

/// Internal: Attention-based aggregation
fn attention_mil_impl(instances: &ArrayView2<f32>, attention_weights: &ArrayView2<f32>) -> Array1<f32> {
    let n_instances = instances.shape()[0];
    let feature_dim = instances.shape()[1];

    // Flatten attention weights if needed
    let weights = if attention_weights.shape()[0] == n_instances && attention_weights.shape()[1] == 1 {
        attention_weights.column(0).to_owned()
    } else if attention_weights.shape()[0] == 1 && attention_weights.shape()[1] == n_instances {
        attention_weights.row(0).to_owned()
    } else if attention_weights.shape()[0] == n_instances {
        attention_weights.column(0).to_owned()
    } else {
        // Assume it's already 1D
        attention_weights.row(0).to_owned()
    };

    // Normalize weights
    let weight_sum: f32 = weights.sum();
    let normalized_weights = if weight_sum > 1e-8 {
        weights.mapv(|w| w / weight_sum)
    } else {
        Array1::from_elem(n_instances, 1.0 / n_instances as f32)
    };

    // Weighted sum
    let mut result = Array1::<f32>::zeros(feature_dim);
    for i in 0..n_instances {
        let weight = normalized_weights[i];
        let instance = instances.row(i);
        for j in 0..feature_dim {
            result[j] += weight * instance[j];
        }
    }

    result
}

/// Internal: Max pooling aggregation
fn max_pooling_mil_impl(instances: &ArrayView2<f32>) -> Array1<f32> {
    instances
        .axis_iter(Axis(1))
        .map(|col| {
            col.iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect()
}

/// Internal: Mean pooling aggregation
fn mean_pooling_mil_impl(instances: &ArrayView2<f32>) -> Array1<f32> {
    instances
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(instances.shape()[1]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_max_pooling() {
        let instances = Array2::from_shape_vec((3, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 1.0, 2.0, 3.0,
            2.0, 4.0, 1.0, 5.0,
        ]).unwrap();

        let result = max_pooling_mil_impl(&instances.view());

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 5.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 5.0);
    }

    #[test]
    fn test_mean_pooling() {
        let instances = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let result = mean_pooling_mil_impl(&instances.view());

        assert_eq!(result.len(), 2);
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
    }
}
