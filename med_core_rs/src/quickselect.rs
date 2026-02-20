/*!
Optimized percentile calculation using quickselect algorithm.

This provides O(n) average time complexity instead of O(n log n) from sorting.
*/

use std::cmp::Ordering;

/// Find the k-th smallest element using quickselect algorithm
///
/// This is much faster than sorting for finding percentiles.
/// Average time complexity: O(n), worst case: O(n²)
fn quickselect(arr: &mut [f32], k: usize) -> f32 {
    if arr.len() == 1 {
        return arr[0];
    }

    let pivot_idx = partition(arr);

    match k.cmp(&pivot_idx) {
        Ordering::Equal => arr[pivot_idx],
        Ordering::Less => quickselect(&mut arr[..pivot_idx], k),
        Ordering::Greater => quickselect(&mut arr[pivot_idx + 1..], k - pivot_idx - 1),
    }
}

/// Partition array around a pivot (Hoare partition scheme)
fn partition(arr: &mut [f32]) -> usize {
    let len = arr.len();
    let pivot_idx = len / 2;
    let pivot = arr[pivot_idx];

    arr.swap(pivot_idx, len - 1);

    let mut i = 0;
    for j in 0..len - 1 {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, len - 1);
    i
}

/// Calculate percentile using quickselect (O(n) average)
pub fn percentile_fast(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let mut data_copy: Vec<f32> = data.to_vec();
    let n = data_copy.len();
    let idx = ((p / 100.0) * (n - 1) as f32).round() as usize;
    let idx = idx.min(n - 1);

    quickselect(&mut data_copy, idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_fast() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test median (50th percentile)
        let p50 = percentile_fast(&data, 50.0);
        assert!((p50 - 5.5).abs() < 0.6); // Should be around 5.5

        // Test 90th percentile
        let p90 = percentile_fast(&data, 90.0);
        assert!((p90 - 9.0).abs() < 1.0); // Should be around 9

        // Test 10th percentile
        let p10 = percentile_fast(&data, 10.0);
        assert!((p10 - 2.0).abs() < 1.0); // Should be around 2
    }

    #[test]
    fn test_quickselect() {
        let mut data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        // Find median (4th smallest in 8 elements)
        let median = quickselect(&mut data, 3);
        assert!((median - 3.0).abs() < 0.1 || (median - 4.0).abs() < 0.1);
    }
}
