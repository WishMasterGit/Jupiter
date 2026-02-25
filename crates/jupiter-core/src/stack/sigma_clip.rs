use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::consts::{EPSILON, PARALLEL_PIXEL_THRESHOLD};
use crate::error::{JupiterError, Result};
use crate::frame::Frame;

/// Parameters for sigma-clipped mean stacking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SigmaClipParams {
    /// Number of rejection iterations (default: 2).
    pub iterations: usize,
    /// Sigma threshold — values beyond mean +/- sigma*threshold are rejected (default: 2.5).
    pub sigma: f32,
}

impl Default for SigmaClipParams {
    fn default() -> Self {
        Self {
            iterations: 2,
            sigma: 2.5,
        }
    }
}

/// Stack frames using sigma-clipped mean.
///
/// Per pixel: compute mean and stddev, reject values more than `sigma` standard
/// deviations from the mean, then recompute the mean from remaining values.
/// Repeat for the configured number of iterations.
/// Parallelizes at the row level for images >= 256x256.
pub fn sigma_clip_stack(frames: &[Frame], params: &SigmaClipParams) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len();

    if h * w >= PARALLEL_PIXEL_THRESHOLD && n > 1 {
        sigma_clip_stack_parallel(frames, params, h, w, n)
    } else {
        sigma_clip_stack_sequential(frames, params, h, w, n)
    }
}

fn sigma_clip_stack_parallel(
    frames: &[Frame],
    params: &SigmaClipParams,
    h: usize,
    w: usize,
    n: usize,
) -> Result<Frame> {
    // Row-parallel: each row allocates its own pixel_values and mask
    let rows: Vec<Vec<f32>> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut pixel_values = vec![0.0f32; n];
            let mut mask = vec![true; n];
            let mut row_result = vec![0.0f32; w];
            for (col, result) in row_result.iter_mut().enumerate() {
                sigma_clip_pixel(
                    frames,
                    row,
                    col,
                    n,
                    params,
                    &mut pixel_values,
                    &mut mask,
                    result,
                );
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
}

fn sigma_clip_stack_sequential(
    frames: &[Frame],
    params: &SigmaClipParams,
    h: usize,
    w: usize,
    n: usize,
) -> Result<Frame> {
    // Sequential for small images
    let mut result = Array2::<f32>::zeros((h, w));
    let mut pixel_values = vec![0.0f32; n];
    let mut mask = vec![true; n];

    for row in 0..h {
        for col in 0..w {
            sigma_clip_pixel(
                frames,
                row,
                col,
                n,
                params,
                &mut pixel_values,
                &mut mask,
                &mut result[[row, col]],
            );
        }
    }
    Ok(Frame::new(result, frames[0].original_bit_depth))
}

#[allow(clippy::too_many_arguments)]
fn sigma_clip_pixel(
    frames: &[Frame],
    row: usize,
    col: usize,
    n: usize,
    params: &SigmaClipParams,
    pixel_values: &mut [f32],
    mask: &mut [bool],
    out: &mut f32,
) {
    for (i, frame) in frames.iter().enumerate() {
        pixel_values[i] = frame.data[[row, col]];
        mask[i] = true;
    }

    for _ in 0..params.iterations {
        let (mean, stddev) = mean_stddev(pixel_values, mask);
        if stddev < EPSILON {
            break;
        }
        let lo = mean - params.sigma * stddev;
        let hi = mean + params.sigma * stddev;
        for i in 0..n {
            if mask[i] && (pixel_values[i] < lo || pixel_values[i] > hi) {
                mask[i] = false;
            }
        }
    }

    let mut sum = 0.0f32;
    let mut count = 0u32;
    for i in 0..n {
        if mask[i] {
            sum += pixel_values[i];
            count += 1;
        }
    }

    *out = if count > 0 {
        sum / count as f32
    } else {
        pixel_values.iter().sum::<f32>() / n as f32
    };
}

fn mean_stddev(values: &[f32], mask: &[bool]) -> (f32, f32) {
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
