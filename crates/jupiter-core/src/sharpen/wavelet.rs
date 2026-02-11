use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::frame::Frame;

/// Parameters for wavelet sharpening.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WaveletParams {
    /// Number of wavelet decomposition layers (typically 6).
    pub num_layers: usize,
    /// Coefficient per layer: >1.0 sharpens, <1.0 suppresses, 1.0 unchanged.
    pub coefficients: Vec<f32>,
    /// Per-layer denoise threshold (soft-thresholding). 0.0 = no denoise.
    /// Small coefficients in each detail layer below threshold are zeroed out.
    #[serde(default)]
    pub denoise: Vec<f32>,
}

impl Default for WaveletParams {
    fn default() -> Self {
        Self {
            num_layers: 6,
            coefficients: vec![1.5, 1.3, 1.2, 1.1, 1.0, 1.0],
            denoise: vec![],
        }
    }
}

/// B3 spline 1D kernel coefficients: [1, 4, 6, 4, 1] / 16.
const B3_KERNEL: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];

/// Decompose an image into wavelet detail layers + residual.
///
/// Returns (detail_layers, residual) where the original can be reconstructed as:
/// original = sum(detail_layers) + residual
pub fn decompose(data: &Array2<f32>, num_layers: usize) -> (Vec<Array2<f32>>, Array2<f32>) {
    let mut layers = Vec::with_capacity(num_layers);
    let mut current = data.clone();

    for scale in 0..num_layers {
        let smoothed = atrous_convolve(&current, scale);
        let detail = &current - &smoothed;
        layers.push(detail);
        current = smoothed;
    }

    (layers, current)
}

/// Reconstruct an image from wavelet layers with given coefficients and denoise thresholds.
pub fn reconstruct(
    layers: &[Array2<f32>],
    residual: &Array2<f32>,
    coefficients: &[f32],
    denoise: &[f32],
) -> Array2<f32> {
    let mut result = residual.clone();

    for (i, layer) in layers.iter().enumerate() {
        let coeff = coefficients.get(i).copied().unwrap_or(1.0);
        let threshold = denoise.get(i).copied().unwrap_or(0.0);

        if threshold > 0.0 {
            // Soft-thresholding: sign(w) * max(0, |w| - threshold)
            let denoised = layer.mapv(|w| {
                let abs_w = w.abs();
                if abs_w <= threshold {
                    0.0
                } else {
                    w.signum() * (abs_w - threshold)
                }
            });
            result += &(denoised * coeff);
        } else {
            result += &(layer * coeff);
        }
    }

    // Clamp to valid range
    result.mapv_inplace(|v| v.clamp(0.0, 1.0));
    result
}

/// Sharpen a frame using a trous wavelet decomposition.
pub fn sharpen(frame: &Frame, params: &WaveletParams) -> Frame {
    let (layers, residual) = decompose(&frame.data, params.num_layers);
    let sharpened = reconstruct(&layers, &residual, &params.coefficients, &params.denoise);
    Frame::new(sharpened, frame.original_bit_depth)
}

/// A trous convolution at a given scale.
///
/// The B3 spline kernel is applied separably (rows then columns)
/// with a dilation factor of 2^scale (reading input at intervals of 2^scale).
fn atrous_convolve(data: &Array2<f32>, scale: usize) -> Array2<f32> {
    let step = 1usize << scale; // 2^scale

    // Convolve rows first
    let row_convolved = convolve_rows(data, &B3_KERNEL, step);
    // Then convolve columns
    convolve_cols(&row_convolved, &B3_KERNEL, step)
}

fn convolve_rows(data: &Array2<f32>, kernel: &[f32; 5], step: usize) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));
    let half = 2; // kernel radius = 2 for 5-tap kernel

    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let offset = (ki as isize - half as isize) * step as isize;
                let src_col = col as isize + offset;
                // Mirror boundary
                let src_col = mirror_index(src_col, w);
                sum += data[[row, src_col]] * kv;
            }
            result[[row, col]] = sum;
        }
    }

    result
}

fn convolve_cols(data: &Array2<f32>, kernel: &[f32; 5], step: usize) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));
    let half = 2;

    for row in 0..h {
        for col in 0..w {
            let mut sum = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let offset = (ki as isize - half as isize) * step as isize;
                let src_row = row as isize + offset;
                let src_row = mirror_index(src_row, h);
                sum += data[[src_row, col]] * kv;
            }
            result[[row, col]] = sum;
        }
    }

    result
}

/// Mirror boundary handling: reflect index into [0, size).
/// Even function (f(-k) = f(k)) with period 2*size, ping-ponging within [0, size).
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
