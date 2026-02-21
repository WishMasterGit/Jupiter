use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::compute::ComputeBackend;
use crate::consts::PARALLEL_FRAME_THRESHOLD;
use crate::error::{JupiterError, Result};
use crate::frame::{AlignmentOffset, Frame};
use crate::io::ser::SerReader;
use crate::pipeline::config::{AlignmentConfig, AlignmentMethod};

use super::phase_correlation;
use super::{centroid, enhanced_phase, gradient_correlation, pyramid, shift_frame};

/// Compute alignment offset between two arrays using the configured method.
pub fn compute_offset_configured(
    reference: &ndarray::Array2<f32>,
    target: &ndarray::Array2<f32>,
    config: &AlignmentConfig,
    backend: &dyn ComputeBackend,
) -> Result<AlignmentOffset> {
    match &config.method {
        AlignmentMethod::PhaseCorrelation => {
            if backend.is_gpu() {
                phase_correlation::compute_offset_gpu(reference, target, backend)
            } else {
                phase_correlation::compute_offset_array(reference, target)
            }
        }
        AlignmentMethod::EnhancedPhaseCorrelation(params) => {
            enhanced_phase::compute_offset_enhanced(reference, target, params, backend)
        }
        AlignmentMethod::Centroid(params) => {
            centroid::compute_offset_centroid(reference, target, params)
        }
        AlignmentMethod::GradientCorrelation => {
            gradient_correlation::compute_offset_gradient(reference, target, backend)
        }
        AlignmentMethod::Pyramid(params) => {
            pyramid::compute_offset_pyramid(reference, target, params, backend)
        }
    }
}

/// Align frames using the configured alignment method with progress reporting.
pub fn align_frames_configured_with_progress<F>(
    frames: &[Frame],
    reference_idx: usize,
    config: &AlignmentConfig,
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
                    let offset = compute_offset_configured(
                        &reference.data,
                        &frame.data,
                        config,
                        backend.as_ref(),
                    )?;
                    if backend.is_gpu() {
                        let shifted_buf = backend.shift_bilinear(
                            &backend.upload(&frame.data),
                            offset.dx,
                            offset.dy,
                        );
                        let shifted_data = backend.download(&shifted_buf);
                        Ok(Frame::new(shifted_data, frame.original_bit_depth))
                    } else {
                        Ok(shift_frame(frame, &offset))
                    }
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
                let offset = compute_offset_configured(
                    &reference.data,
                    &frame.data,
                    config,
                    backend.as_ref(),
                )?;
                if backend.is_gpu() {
                    let shifted_buf = backend.shift_bilinear(
                        &backend.upload(&frame.data),
                        offset.dx,
                        offset.dy,
                    );
                    let shifted_data = backend.download(&shifted_buf);
                    Frame::new(shifted_data, frame.original_bit_depth)
                } else {
                    shift_frame(frame, &offset)
                }
            };
            aligned.push(result);
            let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
            on_frame_done(done);
        }
        Ok(aligned)
    }
}

/// Compute alignment offsets by streaming frames from the SER reader,
/// using the configured alignment method.
pub fn compute_offsets_streaming_configured<F>(
    reader: &SerReader,
    frame_indices: &[usize],
    reference_idx: usize,
    config: &AlignmentConfig,
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

    let results: Vec<Result<AlignmentOffset>> =
        if frame_indices.len() >= PARALLEL_FRAME_THRESHOLD {
            frame_indices
                .par_iter()
                .enumerate()
                .map(|(i, &frame_idx)| {
                    let offset = if i == reference_idx {
                        AlignmentOffset::default()
                    } else {
                        let target = reader.read_frame(frame_idx)?;
                        compute_offset_configured(
                            &reference.data,
                            &target.data,
                            config,
                            backend.as_ref(),
                        )?
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
                        compute_offset_configured(
                            &reference.data,
                            &target.data,
                            config,
                            backend.as_ref(),
                        )?
                    };
                    let done = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    on_frame_done(done);
                    Ok(offset)
                })
                .collect()
        };

    results.into_iter().collect()
}
