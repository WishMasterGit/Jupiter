use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::PARALLEL_PIXEL_THRESHOLD;

/// Collect row vectors into an `Array2<f32>`.
///
/// Each inner `Vec` must have exactly `w` elements.
pub fn rows_to_array2(rows: Vec<Vec<f32>>, h: usize, w: usize) -> Array2<f32> {
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((h, w), flat).expect("row dimensions must match")
}

/// Process rows in parallel or sequentially based on image size,
/// returning the result as an `Array2<f32>`.
///
/// Uses Rayon parallelism when `h * w >= PARALLEL_PIXEL_THRESHOLD`.
pub fn par_or_seq_rows<F>(h: usize, w: usize, row_fn: F) -> Array2<f32>
where
    F: Fn(usize) -> Vec<f32> + Send + Sync,
{
    let rows: Vec<Vec<f32>> = if h * w >= PARALLEL_PIXEL_THRESHOLD {
        (0..h).into_par_iter().map(&row_fn).collect()
    } else {
        (0..h).map(&row_fn).collect()
    };
    rows_to_array2(rows, h, w)
}

/// Compute the median of `pixel_values[..n]` in-place using `select_nth_unstable`.
///
/// For even `n`, returns the average of the two middle values.
pub fn compute_median(pixel_values: &mut [f32], n: usize) -> f32 {
    if n == 1 {
        pixel_values[0]
    } else if n % 2 == 1 {
        let mid = n / 2;
        *pixel_values
            .select_nth_unstable_by(mid, |a, b| a.total_cmp(b))
            .1
    } else {
        let mid = n / 2;
        pixel_values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        pixel_values[..mid].select_nth_unstable_by(mid - 1, |a, b| a.total_cmp(b));
        (pixel_values[mid - 1] + pixel_values[mid]) / 2.0
    }
}

/// Compute masked mean and standard deviation.
///
/// Only elements where `mask[i]` is `true` are included.
/// Returns `(0.0, 0.0)` if no elements are unmasked.
pub fn masked_mean_stddev(values: &[f32], mask: &[bool]) -> (f32, f32) {
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            sum += v;
            count += 1;
        }
    }
    if count == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / count as f32;

    let mut var_sum = 0.0f32;
    for (i, &v) in values.iter().enumerate() {
        if mask[i] {
            let d = v - mean;
            var_sum += d * d;
        }
    }
    let stddev = (var_sum / count as f32).sqrt();
    (mean, stddev)
}

/// Mirror boundary handling: reflect index into `[0, size)`.
///
/// Even function (`f(-k) = f(k)`) with period `2*size`, ping-ponging within `[0, size)`.
pub fn mirror_index(idx: isize, size: usize) -> usize {
    if size <= 1 {
        return 0;
    }
    let period = 2 * size;
    let abs_idx = idx.unsigned_abs();
    let m = abs_idx % period;

    if m < size {
        m
    } else {
        2 * size - 1 - m
    }
}
