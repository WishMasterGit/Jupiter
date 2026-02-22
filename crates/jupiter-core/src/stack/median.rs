use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::PARALLEL_PIXEL_THRESHOLD;
use crate::error::{JupiterError, Result};
use crate::frame::Frame;

/// Stack frames by computing the median at each pixel position.
///
/// Uses `select_nth_unstable` for O(n) median without full sort.
/// Parallelizes at the row level for images >= 256x256.
pub fn median_stack(frames: &[Frame]) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len();

    if h * w >= PARALLEL_PIXEL_THRESHOLD && n > 1 {
        // Row-parallel: each row allocates its own pixel_values
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|row| {
                let mut pixel_values = vec![0.0f32; n];
                let mut row_result = vec![0.0f32; w];
                for (col, result) in row_result.iter_mut().enumerate() {
                    for (i, frame) in frames.iter().enumerate() {
                        pixel_values[i] = frame.data[[row, col]];
                    }
                    *result = compute_median(&mut pixel_values, n);
                }
                row_result
            })
            .collect();

        let mut result = Array2::<f32>::zeros((h, w));
        for (row, row_data) in rows.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                result[[row, col]] = val;
            }
        }
        Ok(Frame::new(result, frames[0].original_bit_depth))
    } else {
        // Sequential for small images
        let mut result = Array2::<f32>::zeros((h, w));
        let mut pixel_values = vec![0.0f32; n];

        for row in 0..h {
            for col in 0..w {
                for (i, frame) in frames.iter().enumerate() {
                    pixel_values[i] = frame.data[[row, col]];
                }
                result[[row, col]] = compute_median(&mut pixel_values, n);
            }
        }
        Ok(Frame::new(result, frames[0].original_bit_depth))
    }
}

fn compute_median(pixel_values: &mut [f32], n: usize) -> f32 {
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
        pixel_values[..mid]
            .select_nth_unstable_by(mid - 1, |a, b| a.total_cmp(b));
        (pixel_values[mid - 1] + pixel_values[mid]) / 2.0
    }
}
