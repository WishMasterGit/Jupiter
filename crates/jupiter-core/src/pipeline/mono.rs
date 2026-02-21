use std::sync::Arc;

use tracing::info;

use crate::align::{
    align_frames_configured_with_progress, compute_offsets_streaming_configured, shift_frame,
};
use crate::compute::ComputeBackend;
use crate::error::Result;
use crate::frame::Frame;
use crate::io::image_io::save_image;
use crate::io::ser::SerReader;
use crate::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use crate::sharpen::wavelet;
use crate::stack::drizzle::{drizzle_stack_streaming, DrizzleConfig};
use crate::stack::mean::StreamingMeanStacker;

use super::config::PipelineConfig;
use super::config::StackMethod;
use super::helpers::{
    apply_filter_step, drizzle_flow, rank_by_metric, rank_by_metric_streaming, select_frames,
    stack_frames_with_progress,
};
use super::types::{PipelineOutput, PipelineStage, ProgressReporter};

/// The existing mono pipeline path (unchanged logic).
pub(super) fn run_mono_pipeline(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<PipelineOutput> {
    let streaming = super::orchestrator::should_use_streaming(reader, config, false);
    if streaming {
        info!("Using low-memory streaming mode");
    }

    let stacked = if let StackMethod::Drizzle(ref drizzle_config) = config.stacking.method {
        if streaming {
            run_mono_drizzle_streaming(reader, config, backend, reporter, drizzle_config, total)?
        } else {
            run_mono_drizzle(reader, config, backend, reporter, drizzle_config, total)?
        }
    } else if streaming {
        run_mono_standard_streaming(reader, config, backend, reporter, total)?
    } else {
        run_mono_standard(reader, config, backend, reporter, total)?
    };

    let output = apply_post_stack_mono(stacked, config, backend, reporter)?;
    Ok(output)
}

fn run_mono_standard(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<Frame> {
    // Read
    reporter.begin_stage(PipelineStage::Reading, Some(total));
    let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
    reporter.finish_stage();

    // Quality
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric(&frames, &config.frame_selection.metric);
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, _) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_frames: Vec<Frame> = selected_indices
        .iter()
        .map(|&i| frames[i].clone())
        .collect();
    info!(
        selected = selected_frames.len(),
        total, "Selected best frames"
    );
    reporter.finish_stage();

    // Alignment
    let frame_count = selected_frames.len();
    reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
    let aligned = if frame_count > 1 {
        let r = reporter.clone();
        align_frames_configured_with_progress(
            &selected_frames,
            0,
            &config.alignment,
            backend.clone(),
            move |done| {
                r.advance(done);
            },
        )?
    } else {
        selected_frames
    };
    reporter.finish_stage();

    // Stacking
    let frame_count = aligned.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
    let r = reporter.clone();
    let result = stack_frames_with_progress(&aligned, &config.stacking.method, move |done| {
        r.advance(done);
    })?;
    info!(method = ?config.stacking.method, "Stacking complete");
    reporter.finish_stage();

    Ok(result)
}

/// Streaming mono pipeline: score -> select -> load-shift-stack one at a time.
///
/// For Mean: fully streaming -- each frame is loaded, shifted, accumulated, then dropped.
/// For Median/SigmaClip: semi-streaming -- offsets computed streaming, then M selected
/// frames loaded+shifted for the per-pixel stacking pass.
fn run_mono_standard_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<Frame> {
    // Quality (streaming: one batch at a time)
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric_streaming(reader, &config.frame_selection.metric)?;
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, _) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    info!(
        selected = selected_indices.len(),
        total, "Selected best frames (streaming)"
    );
    reporter.finish_stage();

    let frame_count = selected_indices.len();

    match &config.stacking.method {
        StackMethod::Mean => {
            // Fully streaming: compute offsets, then load-shift-accumulate one at a time
            reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
            let r = reporter.clone();
            let offsets = compute_offsets_streaming_configured(
                reader,
                &selected_indices,
                0,
                &config.alignment,
                backend.clone(),
                move |done| {
                    r.advance(done);
                },
            )?;
            reporter.finish_stage();

            reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
            let h = reader.header.height as usize;
            let w = reader.header.width as usize;
            let bit_depth = reader.header.pixel_depth as u8;
            let mut stacker = StreamingMeanStacker::new(h, w, bit_depth);
            for (i, (&frame_idx, offset)) in selected_indices.iter().zip(offsets.iter()).enumerate()
            {
                let frame = reader.read_frame(frame_idx)?;
                let shifted = if i == 0 {
                    frame
                } else {
                    shift_frame(&frame, offset)
                };
                stacker.add(&shifted);
                reporter.advance(i + 1);
                // frame + shifted dropped here
            }
            let result = stacker.finalize()?;
            info!(method = "Mean", "Streaming stacking complete");
            reporter.finish_stage();
            Ok(result)
        }
        _ => {
            // Median/SigmaClip: compute offsets streaming, then load M selected + shift
            reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
            let r = reporter.clone();
            let offsets = compute_offsets_streaming_configured(
                reader,
                &selected_indices,
                0,
                &config.alignment,
                backend.clone(),
                move |done| {
                    r.advance(done);
                },
            )?;
            reporter.finish_stage();

            // Load and shift selected frames
            reporter.begin_stage(PipelineStage::Reading, Some(frame_count));
            let mut aligned = Vec::with_capacity(frame_count);
            for (i, (&frame_idx, offset)) in selected_indices.iter().zip(offsets.iter()).enumerate()
            {
                let frame = reader.read_frame(frame_idx)?;
                let shifted = if i == 0 {
                    frame
                } else {
                    shift_frame(&frame, offset)
                };
                aligned.push(shifted);
                reporter.advance(i + 1);
            }
            reporter.finish_stage();

            // Stack
            let stack_count = aligned.len();
            reporter.begin_stage(PipelineStage::Stacking, Some(stack_count));
            let r = reporter.clone();
            let result =
                stack_frames_with_progress(&aligned, &config.stacking.method, move |done| {
                    r.advance(done);
                })?;
            info!(method = ?config.stacking.method, "Streaming stacking complete");
            reporter.finish_stage();
            Ok(result)
        }
    }
}

/// Streaming mono drizzle: score -> select -> stream offsets -> stream drizzle.
fn run_mono_drizzle_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &DrizzleConfig,
    total: usize,
) -> Result<Frame> {
    // Quality (streaming)
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric_streaming(reader, &config.frame_selection.metric)?;
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    info!(
        selected = selected_indices.len(),
        total, "Selected best frames for drizzle (streaming)"
    );
    reporter.finish_stage();

    // Alignment offsets (streaming)
    let frame_count = selected_indices.len();
    reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
    let r = reporter.clone();
    let offsets = compute_offsets_streaming_configured(
        reader,
        &selected_indices,
        0,
        &config.alignment,
        backend.clone(),
        move |done| {
            r.advance(done);
        },
    )?;
    info!("Alignment offsets computed for drizzle (streaming)");
    reporter.finish_stage();

    // Drizzle (streaming: one frame at a time into accumulator)
    reporter.begin_stage(PipelineStage::Stacking, None);
    let scores = if drizzle_config.quality_weighted {
        Some(quality_scores.as_slice())
    } else {
        None
    };
    let result =
        drizzle_stack_streaming(reader, &selected_indices, &offsets, drizzle_config, scores)?;
    info!(
        method = "Drizzle",
        scale = drizzle_config.scale,
        pixfrac = drizzle_config.pixfrac,
        "Streaming drizzle stacking complete"
    );
    reporter.finish_stage();
    Ok(result)
}

fn run_mono_drizzle(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &DrizzleConfig,
    total: usize,
) -> Result<Frame> {
    // Read
    reporter.begin_stage(PipelineStage::Reading, Some(total));
    let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
    reporter.finish_stage();

    // Quality + selection + offsets + drizzle
    drizzle_flow(&frames, config, backend, reporter, drizzle_config, total)
}

/// Post-stacking processing for mono path: sharpen -> filter -> write -> return.
pub(super) fn apply_post_stack_mono(
    stacked: Frame,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<PipelineOutput> {
    // Sharpening
    let mut result = if let Some(ref sharpening_config) = config.sharpening {
        reporter.begin_stage(PipelineStage::Sharpening, None);
        let mut sharpened = stacked;
        if let Some(ref deconv_config) = sharpening_config.deconvolution {
            if backend.is_gpu() {
                sharpened = deconvolve_gpu(&sharpened, deconv_config, &**backend);
            } else {
                sharpened = deconvolve(&sharpened, deconv_config);
            }
            info!("Deconvolution complete");
        }
        sharpened = wavelet::sharpen(&sharpened, &sharpening_config.wavelet);
        info!("Wavelet sharpening complete");
        reporter.finish_stage();
        sharpened
    } else {
        stacked
    };

    // Filters
    if !config.filters.is_empty() {
        let total_filters = config.filters.len();
        reporter.begin_stage(PipelineStage::Filtering, Some(total_filters));
        for (i, step) in config.filters.iter().enumerate() {
            result = apply_filter_step(&result, step);
            reporter.advance(i + 1);
        }
        info!(count = total_filters, "Filters applied");
        reporter.finish_stage();
    }

    // Write
    reporter.begin_stage(PipelineStage::Writing, None);
    save_image(&result, &config.output)?;
    info!(output = %config.output.display(), "Output saved");
    reporter.finish_stage();

    Ok(PipelineOutput::Mono(result))
}
