use ndarray::Array2;
use serde::{Deserialize, Serialize};

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
pub fn sigma_clip_stack(frames: &[Frame], params: &SigmaClipParams) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len();
    let mut result = Array2::<f32>::zeros((h, w));

    let mut pixel_values = vec![0.0f32; n];
    let mut mask = vec![true; n];

    for row in 0..h {
        for col in 0..w {
            for (i, frame) in frames.iter().enumerate() {
                pixel_values[i] = frame.data[[row, col]];
                mask[i] = true;
            }

            for _ in 0..params.iterations {
                let (mean, stddev) = mean_stddev(&pixel_values, &mask);
                if stddev < 1e-10 {
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

            // Compute final mean from surviving values
            let mut sum = 0.0f32;
            let mut count = 0u32;
            for i in 0..n {
                if mask[i] {
                    sum += pixel_values[i];
                    count += 1;
                }
            }

            result[[row, col]] = if count > 0 {
                sum / count as f32
            } else {
                // If all values rejected, fall back to full mean
                pixel_values.iter().sum::<f32>() / n as f32
            };
        }
    }

    Ok(Frame::new(result, frames[0].original_bit_depth))
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
