use ndarray::Array2;
use num_complex::Complex;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::compute::cpu::{fft2d_forward, ifft2d_inverse};
use crate::compute::ComputeBackend;
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, Frame};
use crate::io::ser::SerReader;

use crate::consts::{PARALLEL_FRAME_THRESHOLD, PARALLEL_PIXEL_THRESHOLD};

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
    let ref_fft = fft2d_forward(&ref_windowed);
    let tgt_fft = fft2d_forward(&tgt_windowed);

    // Normalized cross-power spectrum
    let cross_power = normalized_cross_power(&ref_fft, &tgt_fft);

    // Inverse 2D FFT to get correlation surface
    let correlation = ifft2d_inverse(&cross_power);

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

/// Compute offset and a confidence metric (peak_value / mean_correlation).
///
/// Returns `(offset, confidence)`.  A confidence below
/// [`crate::consts::MIN_CORRELATION_CONFIDENCE`] typically indicates an
/// unreliable alignment that should be discarded.
pub fn compute_offset_with_confidence(
    reference: &Array2<f32>,
    target: &Array2<f32>,
) -> Result<(AlignmentOffset, f64)> {
    let (h, w) = reference.dim();
    let (th, tw) = target.dim();
    if h != th || w != tw {
        return Err(JupiterError::Pipeline(format!(
            "Array size mismatch: {}x{} vs {}x{}",
            w, h, tw, th
        )));
    }

    let ref_windowed = apply_hann(reference);
    let tgt_windowed = apply_hann(target);

    let ref_fft = fft2d_forward(&ref_windowed);
    let tgt_fft = fft2d_forward(&tgt_windowed);

    let cross_power = normalized_cross_power(&ref_fft, &tgt_fft);
    let correlation = ifft2d_inverse(&cross_power);

    let (peak_row, peak_col, peak_val) = find_peak(&correlation);

    // Confidence = peak / mean(abs(correlation))
    let n = (h * w) as f64;
    let mean_abs: f64 = correlation.iter().map(|v| v.abs()).sum::<f64>() / n;
    let confidence = if mean_abs > 1e-15 {
        peak_val / mean_abs
    } else {
        0.0
    };

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

    let (sub_dy, sub_dx) = refine_peak_paraboloid(&correlation, peak_row, peak_col);

    Ok((
        AlignmentOffset {
            dx: dx + sub_dx,
            dy: dy + sub_dy,
        },
        confidence,
    ))
}

/// Compute translation offset using a ComputeBackend (GPU or CPU).
pub fn compute_offset_gpu(
    reference: &Array2<f32>,
    target: &Array2<f32>,
    backend: &dyn ComputeBackend,
) -> Result<AlignmentOffset> {
    let (h, w) = reference.dim();
    let (th, tw) = target.dim();
    if h != th || w != tw {
        return Err(JupiterError::Pipeline(format!(
            "Array size mismatch: {}x{} vs {}x{}",
            w, h, tw, th
        )));
    }

    let ref_buf = backend.upload(reference);
    let tgt_buf = backend.upload(target);

    let ref_hann = backend.hann_window(&ref_buf);
    let tgt_hann = backend.hann_window(&tgt_buf);

    let ref_fft = backend.fft2d(&ref_hann);
    let tgt_fft = backend.fft2d(&tgt_hann);

    let cross_power = backend.cross_power_spectrum(&ref_fft, &tgt_fft);

    // Use padded (power-of-2) dimensions for IFFT so negative-shift peaks
    // (which wrap to high indices in the padded domain) are not cropped away.
    let ph = ref_fft.height;
    let pw = ref_fft.width;
    let correlation_buf = backend.ifft2d_real(&cross_power, ph, pw);

    let (peak_row, peak_col, _peak_val) = backend.find_peak(&correlation_buf);

    // Download correlation for subpixel refinement
    let correlation_f32 = backend.download(&correlation_buf);
    let correlation_f64 = correlation_f32.mapv(|v| v as f64);

    // Wrap-around: peaks beyond half the padded size represent negative shifts
    let dy = if peak_row > ph / 2 {
        peak_row as f64 - ph as f64
    } else {
        peak_row as f64
    };
    let dx = if peak_col > pw / 2 {
        peak_col as f64 - pw as f64
    } else {
        peak_col as f64
    };

    let (sub_dy, sub_dx) = refine_peak_paraboloid(&correlation_f64, peak_row, peak_col);

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

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        // Row-parallel: each row's interpolation is independent
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|row| {
                (0..w)
                    .map(|col| {
                        let src_y = row as f64 - offset.dy;
                        let src_x = col as f64 - offset.dx;
                        bilinear_sample(&frame.data, src_y, src_x)
                    })
                    .collect()
            })
            .collect();

        let mut result = Array2::<f32>::zeros((h, w));
        for (row, row_data) in rows.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                result[[row, col]] = val;
            }
        }
        Frame::new(result, frame.original_bit_depth)
    } else {
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
}

/// Shift a raw array by the given offset using bilinear interpolation.
pub(crate) fn shift_array(data: &Array2<f32>, offset: &AlignmentOffset) -> Array2<f32> {
    let (h, w) = data.dim();

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        let rows: Vec<Vec<f32>> = (0..h)
            .into_par_iter()
            .map(|row| {
                (0..w)
                    .map(|col| {
                        let src_y = row as f64 - offset.dy;
                        let src_x = col as f64 - offset.dx;
                        bilinear_sample(data, src_y, src_x)
                    })
                    .collect()
            })
            .collect();

        let mut result = Array2::<f32>::zeros((h, w));
        for (row, row_data) in rows.into_iter().enumerate() {
            for (col, val) in row_data.into_iter().enumerate() {
                result[[row, col]] = val;
            }
        }
        result
    } else {
        let mut result = Array2::<f32>::zeros((h, w));
        for row in 0..h {
            for col in 0..w {
                let src_y = row as f64 - offset.dy;
                let src_x = col as f64 - offset.dx;
                result[[row, col]] = bilinear_sample(data, src_y, src_x);
            }
        }
        result
    }
}

/// Align a sequence of frames to a reference frame.
///
/// Uses parallel processing when there are enough frames to benefit.
pub fn align_frames(frames: &[Frame], reference_idx: usize) -> Result<Vec<Frame>> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let reference = &frames[reference_idx];

    if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        let results: Vec<Result<Frame>> = frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| {
                if i == reference_idx {
                    Ok(frame.clone())
                } else {
                    let offset = compute_offset(reference, frame)?;
                    Ok(shift_frame(frame, &offset))
                }
            })
            .collect();
        results.into_iter().collect()
    } else {
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
}

/// Align a sequence of frames to a reference frame with progress reporting.
///
/// The `on_frame_done` callback receives the number of frames completed so far.
/// It must be `Fn + Send + Sync` to support parallel execution.
pub fn align_frames_with_progress<F>(
    frames: &[Frame],
    reference_idx: usize,
    on_frame_done: F,
) -> Result<Vec<Frame>>
where
    F: Fn(usize) + Send + Sync,
{
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let reference = &frames[reference_idx];
    let counter = AtomicUsize::new(0);

    let results: Vec<Result<Frame>> = frames
        .par_iter()
        .enumerate()
        .map(|(i, frame)| {
            let result = if i == reference_idx {
                Ok(frame.clone())
            } else {
                let offset = compute_offset(reference, frame)?;
                Ok(shift_frame(frame, &offset))
            };
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            on_frame_done(done);
            result
        })
        .collect();

    results.into_iter().collect()
}

/// Align frames using a ComputeBackend (GPU or CPU) with progress reporting.
pub fn align_frames_gpu_with_progress<F>(
    frames: &[Frame],
    reference_idx: usize,
    backend: Arc<dyn ComputeBackend>,
    on_frame_done: F,
) -> Result<Vec<Frame>>
where
    F: Fn(usize) + Send + Sync,
{
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let reference = &frames[reference_idx];
    let counter = AtomicUsize::new(0);

    if frames.len() >= PARALLEL_FRAME_THRESHOLD {
        let results: Vec<Result<Frame>> = frames
            .par_iter()
            .enumerate()
            .map(|(i, frame)| {
                let result = if i == reference_idx {
                    Ok(frame.clone())
                } else {
                    let offset =
                        compute_offset_gpu(&reference.data, &frame.data, backend.as_ref())?;
                    let shifted_buf = backend.shift_bilinear(
                        &backend.upload(&frame.data),
                        offset.dx,
                        offset.dy,
                    );
                    let shifted_data = backend.download(&shifted_buf);
                    Ok(Frame::new(shifted_data, frame.original_bit_depth))
                };
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                on_frame_done(done);
                result
            })
            .collect();
        results.into_iter().collect()
    } else {
        let mut aligned = Vec::with_capacity(frames.len());
        for (i, frame) in frames.iter().enumerate() {
            let result = if i == reference_idx {
                frame.clone()
            } else {
                let offset =
                    compute_offset_gpu(&reference.data, &frame.data, backend.as_ref())?;
                let shifted_buf =
                    backend.shift_bilinear(&backend.upload(&frame.data), offset.dx, offset.dy);
                let shifted_data = backend.download(&shifted_buf);
                Frame::new(shifted_data, frame.original_bit_depth)
            };
            aligned.push(result);
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            on_frame_done(done);
        }
        Ok(aligned)
    }
}

pub(crate) fn apply_hann(data: &Array2<f32>) -> Array2<f32> {
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

pub(crate) fn normalized_cross_power(
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

pub(crate) fn find_peak(data: &Array2<f64>) -> (usize, usize, f64) {
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

/// Compute alignment offsets by streaming frames from the SER reader.
///
/// Only the reference frame is held in memory persistently. Each target frame
/// is loaded, offset-computed, then dropped. Uses Rayon parallelism when
/// `frame_indices.len() >= PARALLEL_FRAME_THRESHOLD`.
pub fn compute_offsets_streaming<F>(
    reader: &SerReader,
    frame_indices: &[usize],
    reference_idx: usize,
    backend: Arc<dyn ComputeBackend>,
    on_frame_done: F,
) -> Result<Vec<AlignmentOffset>>
where
    F: Fn(usize) + Send + Sync,
{
    if frame_indices.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let reference = reader.read_frame(frame_indices[reference_idx])?;
    let counter = AtomicUsize::new(0);

    let results: Vec<Result<AlignmentOffset>> = if frame_indices.len() >= PARALLEL_FRAME_THRESHOLD {
        frame_indices
            .par_iter()
            .enumerate()
            .map(|(i, &frame_idx)| {
                let offset = if i == reference_idx {
                    AlignmentOffset::default()
                } else {
                    let target = reader.read_frame(frame_idx)?;
                    if backend.is_gpu() {
                        compute_offset_gpu(&reference.data, &target.data, backend.as_ref())?
                    } else {
                        compute_offset_array(&reference.data, &target.data)?
                    }
                    // target dropped here
                };
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                on_frame_done(done);
                Ok(offset)
            })
            .collect()
    } else {
        frame_indices
            .iter()
            .enumerate()
            .map(|(i, &frame_idx)| {
                let offset = if i == reference_idx {
                    AlignmentOffset::default()
                } else {
                    let target = reader.read_frame(frame_idx)?;
                    if backend.is_gpu() {
                        compute_offset_gpu(&reference.data, &target.data, backend.as_ref())?
                    } else {
                        compute_offset_array(&reference.data, &target.data)?
                    }
                };
                let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                on_frame_done(done);
                Ok(offset)
            })
            .collect()
    };

    results.into_iter().collect()
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
