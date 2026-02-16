//! Enhanced phase correlation using matrix-multiply DFT (Guizar-Sicairos et al., 2008).
//!
//! Two-stage approach:
//! 1. **Coarse**: Standard FFT phase correlation for integer-pixel peak.
//! 2. **Fine**: Selective upsampling via matrix-multiply DFT in a small window
//!    around the coarse peak, achieving sub-pixel accuracy of ~1/upsample_factor pixels.
//!
//! Reference: "Efficient subpixel image registration algorithms",
//!            M. Guizar-Sicairos, S. T. Thurman, J. R. Fienup, Optics Letters 33(2), 2008.

use ndarray::Array2;
use num_complex::Complex;
use std::f64::consts::TAU;

use crate::compute::cpu::{fft2d_forward, ifft2d_inverse};
use crate::compute::ComputeBackend;
use crate::consts::ENHANCED_PHASE_SEARCH_WINDOW;
use crate::error::{JupiterError, Result};
use crate::frame::AlignmentOffset;
use crate::pipeline::config::EnhancedPhaseConfig;

use super::phase_correlation::{apply_hann, find_peak, normalized_cross_power};

/// Compute alignment offset using enhanced phase correlation.
///
/// Stage 1 uses standard FFT for coarse peak finding. Stage 2 uses
/// matrix-multiply DFT to refine the peak to ~1/upsample_factor pixel accuracy.
pub fn compute_offset_enhanced(
    reference: &Array2<f32>,
    target: &Array2<f32>,
    config: &EnhancedPhaseConfig,
    _backend: &dyn ComputeBackend,
) -> Result<AlignmentOffset> {
    let (h, w) = reference.dim();
    let (th, tw) = target.dim();
    if h != th || w != tw {
        return Err(JupiterError::Pipeline(format!(
            "Array size mismatch: {}x{} vs {}x{}",
            w, h, tw, th
        )));
    }

    // Stage 1: Coarse alignment via standard FFT phase correlation.
    // We retain the FFTs for stage 2.
    let ref_windowed = apply_hann(reference);
    let tgt_windowed = apply_hann(target);

    let ref_fft = fft2d_forward(&ref_windowed);
    let tgt_fft = fft2d_forward(&tgt_windowed);

    let cross_power = normalized_cross_power(&ref_fft, &tgt_fft);
    let correlation = ifft2d_inverse(&cross_power);

    let (peak_row, peak_col, _) = find_peak(&correlation);

    // Convert to signed offset (handle wrap-around)
    let coarse_dy = if peak_row > h / 2 {
        peak_row as f64 - h as f64
    } else {
        peak_row as f64
    };
    let coarse_dx = if peak_col > w / 2 {
        peak_col as f64 - w as f64
    } else {
        peak_col as f64
    };

    if config.upsample_factor <= 1 {
        return Ok(AlignmentOffset {
            dx: coarse_dx,
            dy: coarse_dy,
        });
    }

    // Stage 2: Upsampled matrix-multiply DFT refinement.
    // Evaluate the cross-correlation at sub-pixel locations around the coarse peak.
    let upsample = config.upsample_factor as f64;
    let window = ENHANCED_PHASE_SEARCH_WINDOW;
    let upsampled_size = (window * upsample).ceil() as usize;

    // Center of the upsampled region in the original frequency domain
    let row_shift = coarse_dy;
    let col_shift = coarse_dx;

    // Build row and column DFT kernel matrices for the upsampled region.
    // row_kernel: (w, upsampled_size) — evaluates DFT at upsampled column positions
    // col_kernel: (upsampled_size, h) — evaluates DFT at upsampled row positions
    let row_kernel = build_dft_kernel(w, upsampled_size, col_shift, upsample);
    let col_kernel = build_dft_kernel(h, upsampled_size, row_shift, upsample);

    // Compute upsampled correlation: col_kernel^H * cross_power * row_kernel
    // cross_power is (h, w), col_kernel is (upsampled_size, h), row_kernel is (w, upsampled_size)
    // Result: (upsampled_size, upsampled_size)
    let upsampled_cc = matrix_multiply_dft(&cross_power, &col_kernel, &row_kernel);

    // Find peak in the upsampled correlation
    let mut best_row = 0;
    let mut best_col = 0;
    let mut best_val = f64::NEG_INFINITY;

    for r in 0..upsampled_size {
        for c in 0..upsampled_size {
            let val = upsampled_cc[[r, c]].norm();
            if val > best_val {
                best_val = val;
                best_row = r;
                best_col = c;
            }
        }
    }

    // Convert upsampled peak indices back to sub-pixel offsets.
    // The upsampled region spans [shift - window/2, shift + window/2] at 1/upsample spacing.
    let refined_dy = row_shift - window / 2.0 + best_row as f64 / upsample;
    let refined_dx = col_shift - window / 2.0 + best_col as f64 / upsample;

    Ok(AlignmentOffset {
        dx: refined_dx,
        dy: refined_dy,
    })
}

/// Build a DFT kernel matrix for evaluating the DFT at upsampled positions.
///
/// For a signal of length `n`, evaluates at `upsampled_size` positions centered
/// around `center_shift` with spacing `1/upsample_factor`.
///
/// Returns a matrix of shape `(n, upsampled_size)` where entry (k, j) is:
///   exp(-i * 2π * freq_k * pos_j / n)
///
/// `freq_k` ranges over [0, 1, ..., n-1] shifted so that DC is at center.
/// `pos_j` ranges over upsampled positions near `center_shift`.
fn build_dft_kernel(
    n: usize,
    upsampled_size: usize,
    center_shift: f64,
    upsample_factor: f64,
) -> Array2<Complex<f64>> {
    let mut kernel = Array2::<Complex<f64>>::zeros((n, upsampled_size));
    let half_n = n as f64 / 2.0;
    let start_pos = center_shift - (upsampled_size as f64 - 1.0) / (2.0 * upsample_factor);

    for k in 0..n {
        // Frequency index centered around DC
        let freq = if (k as f64) <= half_n {
            k as f64
        } else {
            k as f64 - n as f64
        };

        for j in 0..upsampled_size {
            let pos = start_pos + j as f64 / upsample_factor;
            let phase = -TAU * freq * pos / n as f64;
            kernel[[k, j]] = Complex::new(phase.cos(), phase.sin());
        }
    }

    kernel
}

/// Compute the upsampled cross-correlation via matrix-multiply DFT.
///
/// `cross_power`: (h, w) complex cross-power spectrum
/// `col_kernel`: (h, upsampled_size) DFT kernel for rows
/// `row_kernel`: (w, upsampled_size) DFT kernel for columns
///
/// Result: conj(col_kernel)^T * cross_power * row_kernel = (upsampled_size, upsampled_size)
fn matrix_multiply_dft(
    cross_power: &Array2<Complex<f64>>,
    col_kernel: &Array2<Complex<f64>>,
    row_kernel: &Array2<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let (h, w) = cross_power.dim();
    let up_rows = col_kernel.dim().1;
    let up_cols = row_kernel.dim().1;

    // Step 1: intermediate = conj(col_kernel)^T * cross_power → (up_rows, w)
    let mut intermediate = Array2::<Complex<f64>>::zeros((up_rows, w));
    for ur in 0..up_rows {
        for c in 0..w {
            let mut sum = Complex::new(0.0, 0.0);
            for r in 0..h {
                sum += col_kernel[[r, ur]].conj() * cross_power[[r, c]];
            }
            intermediate[[ur, c]] = sum;
        }
    }

    // Step 2: result = intermediate * row_kernel → (up_rows, up_cols)
    let mut result = Array2::<Complex<f64>>::zeros((up_rows, up_cols));
    for ur in 0..up_rows {
        for uc in 0..up_cols {
            let mut sum = Complex::new(0.0, 0.0);
            for c in 0..w {
                sum += intermediate[[ur, c]] * row_kernel[[c, uc]];
            }
            result[[ur, uc]] = sum;
        }
    }

    result
}
