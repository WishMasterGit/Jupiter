use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, Frame};

use super::subpixel::refine_peak_paraboloid;

/// Compute the translation offset between two raw arrays using FFT phase correlation.
pub fn compute_offset_array(
    reference: &Array2<f32>,
    target: &Array2<f32>,
) -> Result<AlignmentOffset> {
    let (h, w) = reference.dim();
    let (th, tw) = target.dim();
    if h != th || w != tw {
        return Err(JupiterError::Pipeline(format!(
            "Array size mismatch: {}x{} vs {}x{}",
            w, h, tw, th
        )));
    }

    // Apply Hann window to reduce spectral leakage
    let ref_windowed = apply_hann(reference);
    let tgt_windowed = apply_hann(target);

    // 2D FFT of both
    let ref_fft = fft2d(&ref_windowed);
    let tgt_fft = fft2d(&tgt_windowed);

    // Normalized cross-power spectrum
    let cross_power = normalized_cross_power(&ref_fft, &tgt_fft);

    // Inverse 2D FFT to get correlation surface
    let correlation = ifft2d(&cross_power);

    // Find peak in the correlation surface
    let (peak_row, peak_col, _peak_val) = find_peak(&correlation);

    // Convert to signed offset (handle wrap-around)
    let dy = if peak_row > h / 2 {
        peak_row as f64 - h as f64
    } else {
        peak_row as f64
    };
    let dx = if peak_col > w / 2 {
        peak_col as f64 - w as f64
    } else {
        peak_col as f64
    };

    // Subpixel refinement
    let (sub_dy, sub_dx) = refine_peak_paraboloid(&correlation, peak_row, peak_col);

    Ok(AlignmentOffset {
        dx: dx + sub_dx,
        dy: dy + sub_dy,
    })
}

/// Compute the translation offset between a reference and target frame
/// using FFT phase correlation.
pub fn compute_offset(reference: &Frame, target: &Frame) -> Result<AlignmentOffset> {
    compute_offset_array(&reference.data, &target.data)
}

/// Shift a frame by the given offset using bilinear interpolation.
pub fn shift_frame(frame: &Frame, offset: &AlignmentOffset) -> Frame {
    let (h, w) = frame.data.dim();
    let mut result = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let src_y = row as f64 - offset.dy;
            let src_x = col as f64 - offset.dx;

            result[[row, col]] = bilinear_sample(&frame.data, src_y, src_x);
        }
    }

    Frame::new(result, frame.original_bit_depth)
}

/// Align a sequence of frames to a reference frame.
pub fn align_frames(frames: &[Frame], reference_idx: usize) -> Result<Vec<Frame>> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let reference = &frames[reference_idx];
    let mut aligned = Vec::with_capacity(frames.len());

    for (i, frame) in frames.iter().enumerate() {
        if i == reference_idx {
            aligned.push(frame.clone());
        } else {
            let offset = compute_offset(reference, frame)?;
            aligned.push(shift_frame(frame, &offset));
        }
    }

    Ok(aligned)
}

fn apply_hann(data: &Array2<f32>) -> Array2<f32> {
    let (h, w) = data.dim();
    let mut result = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        let wy = 0.5 * (1.0 - (std::f64::consts::TAU * row as f64 / h as f64).cos());
        for col in 0..w {
            let wx = 0.5 * (1.0 - (std::f64::consts::TAU * col as f64 / w as f64).cos());
            result[[row, col]] = data[[row, col]] * (wy * wx) as f32;
        }
    }

    result
}

/// 2D FFT: row-wise FFT, then column-wise FFT.
fn fft2d(data: &Array2<f32>) -> Array2<Complex<f64>> {
    let (h, w) = data.dim();
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft_forward(w);
    let fft_col = planner.plan_fft_forward(h);

    // Convert to complex
    let mut result = Array2::<Complex<f64>>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            result[[row, col]] = Complex::new(data[[row, col]] as f64, 0.0);
        }
    }

    // Row-wise FFT
    for row in 0..h {
        let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| result[[row, c]]).collect();
        fft_row.process(&mut row_data);
        for col in 0..w {
            result[[row, col]] = row_data[col];
        }
    }

    // Column-wise FFT
    for col in 0..w {
        let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| result[[r, col]]).collect();
        fft_col.process(&mut col_data);
        for row in 0..h {
            result[[row, col]] = col_data[row];
        }
    }

    result
}

/// Inverse 2D FFT.
fn ifft2d(data: &Array2<Complex<f64>>) -> Array2<f64> {
    let (h, w) = data.dim();
    let mut planner = FftPlanner::new();
    let ifft_row = planner.plan_fft_inverse(w);
    let ifft_col = planner.plan_fft_inverse(h);

    let mut work = data.clone();

    // Column-wise IFFT
    for col in 0..w {
        let mut col_data: Vec<Complex<f64>> = (0..h).map(|r| work[[r, col]]).collect();
        ifft_col.process(&mut col_data);
        for row in 0..h {
            work[[row, col]] = col_data[row];
        }
    }

    // Row-wise IFFT
    for row in 0..h {
        let mut row_data: Vec<Complex<f64>> = (0..w).map(|c| work[[row, c]]).collect();
        ifft_row.process(&mut row_data);
        for col in 0..w {
            work[[row, col]] = row_data[col];
        }
    }

    // Extract real part and normalize
    let scale = 1.0 / (h * w) as f64;
    let mut result = Array2::<f64>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            result[[row, col]] = work[[row, col]].re * scale;
        }
    }

    result
}

fn normalized_cross_power(
    ref_fft: &Array2<Complex<f64>>,
    tgt_fft: &Array2<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let (h, w) = ref_fft.dim();
    let mut result = Array2::<Complex<f64>>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            let cross = ref_fft[[row, col]] * tgt_fft[[row, col]].conj();
            let mag = cross.norm();
            result[[row, col]] = if mag > 1e-12 {
                cross / mag
            } else {
                Complex::new(0.0, 0.0)
            };
        }
    }

    result
}

fn find_peak(data: &Array2<f64>) -> (usize, usize, f64) {
    let (h, w) = data.dim();
    let mut best_row = 0;
    let mut best_col = 0;
    let mut best_val = f64::NEG_INFINITY;

    for row in 0..h {
        for col in 0..w {
            if data[[row, col]] > best_val {
                best_val = data[[row, col]];
                best_row = row;
                best_col = col;
            }
        }
    }

    (best_row, best_col, best_val)
}

pub fn bilinear_sample(data: &Array2<f32>, y: f64, x: f64) -> f32 {
    let (h, w) = data.dim();

    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let sample = |r: i64, c: i64| -> f32 {
        if r >= 0 && r < h as i64 && c >= 0 && c < w as i64 {
            data[[r as usize, c as usize]]
        } else {
            0.0
        }
    };

    let v00 = sample(y0, x0);
    let v10 = sample(y0, x1);
    let v01 = sample(y1, x0);
    let v11 = sample(y1, x1);

    v00 * (1.0 - fx) * (1.0 - fy)
        + v10 * fx * (1.0 - fy)
        + v01 * (1.0 - fx) * fy
        + v11 * fx * fy
}
