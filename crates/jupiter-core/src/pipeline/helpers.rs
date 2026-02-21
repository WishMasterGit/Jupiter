use std::sync::Arc;

use tracing::info;

use crate::align::{compute_offset_configured, shift_frame};
use crate::compute::ComputeBackend;
use crate::error::Result;
use crate::filters::gaussian_blur::gaussian_blur;
use crate::filters::histogram::{auto_stretch, histogram_stretch};
use crate::filters::levels::{brightness_contrast, gamma_correct};
use crate::filters::unsharp_mask::unsharp_mask;
use crate::frame::{AlignmentOffset, ColorFrame, Frame, QualityScore};
use crate::io::ser::SerReader;
use crate::quality::gradient::{rank_frames_gradient, rank_frames_gradient_streaming};
use crate::quality::laplacian::{rank_frames, rank_frames_streaming};
use crate::stack::drizzle::{drizzle_stack_with_progress, DrizzleConfig};
use crate::stack::mean::mean_stack_with_progress;
use crate::stack::median::median_stack;
use crate::stack::sigma_clip::sigma_clip_stack;

use super::config::{AlignmentConfig, FilterStep, QualityMetric, StackMethod};
use super::types::{PipelineStage, ProgressReporter};

pub(super) fn rank_by_metric(
    frames: &[Frame],
    metric: &QualityMetric,
) -> Vec<(usize, QualityScore)> {
    match metric {
        QualityMetric::Laplacian => rank_frames(frames),
        QualityMetric::Gradient => rank_frames_gradient(frames),
    }
}

/// Streaming variant: score frames one-batch-at-a-time from the SER reader.
pub(super) fn rank_by_metric_streaming(
    reader: &SerReader,
    metric: &QualityMetric,
) -> Result<Vec<(usize, QualityScore)>> {
    match metric {
        QualityMetric::Laplacian => rank_frames_streaming(reader),
        QualityMetric::Gradient => rank_frames_gradient_streaming(reader),
    }
}

pub(super) fn select_frames(
    ranked: &[(usize, QualityScore)],
    total: usize,
    select_percentage: f32,
) -> (Vec<usize>, Vec<f64>) {
    let keep = (total as f32 * select_percentage).ceil() as usize;
    let keep = keep.max(1).min(total);
    let top: Vec<_> = ranked.iter().take(keep).collect();
    let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let scores: Vec<f64> = top.iter().map(|(_, s)| s.composite).collect();
    (indices, scores)
}

pub(super) fn stack_frames_with_progress(
    frames: &[Frame],
    method: &StackMethod,
    on_progress: impl Fn(usize),
) -> Result<Frame> {
    match method {
        StackMethod::Mean => mean_stack_with_progress(frames, on_progress),
        StackMethod::Median => {
            let result = median_stack(frames);
            on_progress(frames.len());
            result
        }
        StackMethod::SigmaClip(params) => {
            let result = sigma_clip_stack(frames, params);
            on_progress(frames.len());
            result
        }
        StackMethod::MultiPoint(_) | StackMethod::Drizzle(_) => {
            unreachable!("multi-point and drizzle handled separately")
        }
    }
}

/// Compute alignment offsets for a set of frames against a reference frame.
///
/// Reports progress per-frame via the reporter and propagates alignment errors.
pub(super) fn compute_offsets_with_progress(
    frames: &[Frame],
    reference_idx: usize,
    alignment_config: &AlignmentConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<Vec<AlignmentOffset>> {
    let frame_count = frames.len();
    reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
    let reference = &frames[reference_idx];
    let mut offsets = Vec::with_capacity(frame_count);
    for (i, frame) in frames.iter().enumerate() {
        let offset = if i == reference_idx {
            AlignmentOffset::default()
        } else {
            compute_offset_configured(
                &reference.data,
                &frame.data,
                alignment_config,
                backend.as_ref(),
            )?
        };
        offsets.push(offset);
        reporter.advance(i + 1);
    }
    reporter.finish_stage();
    Ok(offsets)
}

/// Split a slice of color frames into per-channel frame vectors (R, G, B).
pub(super) fn split_color_channels(
    frames: &[ColorFrame],
) -> (Vec<Frame>, Vec<Frame>, Vec<Frame>) {
    let red: Vec<Frame> = frames.iter().map(|cf| cf.red.clone()).collect();
    let green: Vec<Frame> = frames.iter().map(|cf| cf.green.clone()).collect();
    let blue: Vec<Frame> = frames.iter().map(|cf| cf.blue.clone()).collect();
    (red, green, blue)
}

/// Stack three channel frame lists in parallel using rayon, returning a ColorFrame.
pub(super) fn stack_color_channels_parallel(
    red: &[Frame],
    green: &[Frame],
    blue: &[Frame],
    method: &StackMethod,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<ColorFrame> {
    let r = reporter.clone();
    let (sr, (sg, sb)) = rayon::join(
        || stack_frames_with_progress(red, method, |done| r.advance(done)),
        || {
            rayon::join(
                || stack_frames_with_progress(green, method, |_| {}),
                || stack_frames_with_progress(blue, method, |_| {}),
            )
        },
    );
    Ok(ColorFrame {
        red: sr?,
        green: sg?,
        blue: sb?,
    })
}

/// Drizzle-stack three channel frame lists in parallel, returning a ColorFrame.
pub(super) fn drizzle_color_channels_parallel(
    red: &[Frame],
    green: &[Frame],
    blue: &[Frame],
    offsets: &[AlignmentOffset],
    drizzle_config: &DrizzleConfig,
    scores: Option<&[f64]>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<ColorFrame> {
    let r = reporter.clone();
    let (dr, (dg, db)) = rayon::join(
        || {
            drizzle_stack_with_progress(red, offsets, drizzle_config, scores, move |done| {
                r.advance(done)
            })
        },
        || {
            rayon::join(
                || drizzle_stack_with_progress(green, offsets, drizzle_config, scores, |_| {}),
                || drizzle_stack_with_progress(blue, offsets, drizzle_config, scores, |_| {}),
            )
        },
    );
    Ok(ColorFrame {
        red: dr?,
        green: dg?,
        blue: db?,
    })
}

/// Apply alignment offsets to color frames (shift each R/G/B channel).
pub(super) fn shift_color_frames(
    frames: &[ColorFrame],
    offsets: &[AlignmentOffset],
) -> Vec<ColorFrame> {
    frames
        .iter()
        .zip(offsets.iter())
        .map(|(cf, offset)| ColorFrame {
            red: shift_frame(&cf.red, offset),
            green: shift_frame(&cf.green, offset),
            blue: shift_frame(&cf.blue, offset),
        })
        .collect()
}

pub(super) fn drizzle_flow(
    frames: &[Frame],
    config: &super::config::PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &DrizzleConfig,
    total: usize,
) -> Result<Frame> {
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric(frames, &config.frame_selection.metric);
    reporter.finish_stage();

    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_frames: Vec<Frame> = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();
    info!(
        selected = selected_frames.len(),
        total, "Selected best frames for drizzle"
    );
    reporter.finish_stage();

    // Compute alignment offsets
    let offsets =
        compute_offsets_with_progress(&selected_frames, 0, &config.alignment, backend, reporter)?;
    info!("Alignment offsets computed for drizzle");

    let drizzle_count = selected_frames.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(drizzle_count));
    let scores = if drizzle_config.quality_weighted {
        Some(quality_scores.as_slice())
    } else {
        None
    };
    let r = reporter.clone();
    let result = drizzle_stack_with_progress(
        &selected_frames,
        &offsets,
        drizzle_config,
        scores,
        move |done| {
            r.advance(done);
        },
    )?;
    info!(
        method = "Drizzle",
        scale = drizzle_config.scale,
        pixfrac = drizzle_config.pixfrac,
        "Drizzle stacking complete"
    );
    reporter.finish_stage();
    Ok(result)
}

/// Apply a single filter step to a frame.
pub fn apply_filter_step(frame: &Frame, step: &FilterStep) -> Frame {
    match step {
        FilterStep::HistogramStretch {
            black_point,
            white_point,
        } => histogram_stretch(frame, *black_point, *white_point),
        FilterStep::AutoStretch {
            low_percentile,
            high_percentile,
        } => auto_stretch(frame, *low_percentile, *high_percentile),
        FilterStep::Gamma(gamma) => gamma_correct(frame, *gamma),
        FilterStep::BrightnessContrast {
            brightness,
            contrast,
        } => brightness_contrast(frame, *brightness, *contrast),
        FilterStep::UnsharpMask {
            radius,
            amount,
            threshold,
        } => unsharp_mask(frame, *radius, *amount, *threshold),
        FilterStep::GaussianBlur { sigma } => gaussian_blur(frame, *sigma),
    }
}
