use std::sync::Arc;

use tracing::info;

use crate::color::debayer::{debayer, luminance, DebayerMethod};
use crate::color::process::process_color_parallel;
use crate::compute::ComputeBackend;
use crate::error::Result;
use crate::frame::{ColorFrame, ColorMode, Frame};
use crate::io::image_io::save_color_image;
use crate::io::ser::SerReader;
use crate::quality::gradient::rank_frames_gradient_color_streaming;
use crate::quality::laplacian::rank_frames_color_streaming;
use crate::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use crate::sharpen::wavelet;
use crate::stack::drizzle::DrizzleConfig;

use super::config::{AlignmentConfig, PipelineConfig, QualityMetric, StackMethod};
use super::helpers::{
    apply_filter_step, compute_offsets_with_progress, drizzle_color_channels_parallel,
    rank_by_metric, select_frames, shift_color_frames, split_color_channels,
    stack_color_channels_parallel,
};
use super::orchestrator::should_use_streaming;
use super::types::{PipelineOutput, PipelineStage, ProgressReporter};

pub(super) fn run_color_pipeline(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    debayer_method: &DebayerMethod,
    color_mode: &ColorMode,
    total: usize,
) -> Result<PipelineOutput> {
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);
    let streaming = should_use_streaming(reader, config, true);

    if streaming {
        info!("Using low-memory streaming mode for color");
        return run_color_pipeline_streaming(
            reader,
            config,
            backend,
            reporter,
            debayer_method,
            color_mode,
            total,
        );
    }

    // Read + debayer (or split RGB)
    reporter.begin_stage(PipelineStage::Reading, Some(total));
    let color_frames: Vec<ColorFrame> = if is_rgb_bgr {
        (0..total)
            .map(|i| reader.read_frame_as_color(i, debayer_method))
            .collect::<Result<_>>()?
    } else {
        let raw_frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
        reporter.finish_stage();
        reporter.begin_stage(PipelineStage::Debayering, Some(total));
        raw_frames
            .iter()
            .map(|frame| {
                debayer(
                    &frame.data,
                    color_mode,
                    debayer_method,
                    frame.original_bit_depth,
                )
                .expect("is_bayer should be true here")
            })
            .collect()
    };
    reporter.finish_stage();

    // Compute luminance for quality scoring
    let lum_frames: Vec<Frame> = color_frames.iter().map(luminance).collect();

    // Quality
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric(&lum_frames, &config.frame_selection.metric);
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_color: Vec<ColorFrame> = selected_indices
        .iter()
        .map(|&i| color_frames[i].clone())
        .collect();
    let selected_lum: Vec<Frame> = selected_indices
        .iter()
        .map(|&i| lum_frames[i].clone())
        .collect();
    info!(
        selected = selected_color.len(),
        total, "Selected best frames (color)"
    );
    reporter.finish_stage();

    let stacked_color = if let StackMethod::Drizzle(ref drizzle_config) = config.stacking.method {
        color_drizzle_flow(
            &selected_color,
            &selected_lum,
            &quality_scores,
            backend,
            reporter,
            drizzle_config,
            &config.alignment,
        )?
    } else {
        color_standard_flow(&selected_color, &selected_lum, config, backend, reporter)?
    };

    apply_post_stack_color(stacked_color, config, backend, reporter)
}

/// Streaming color pipeline: score via batched read-debayer-luminance-score-drop,
/// then re-read only selected frames for stacking.
fn run_color_pipeline_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    debayer_method: &DebayerMethod,
    color_mode: &ColorMode,
    total: usize,
) -> Result<PipelineOutput> {
    // Quality (streaming: read-debayer-luminance-score in batches)
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = match config.frame_selection.metric {
        QualityMetric::Laplacian => {
            rank_frames_color_streaming(reader, color_mode, debayer_method)?
        }
        QualityMetric::Gradient => {
            rank_frames_gradient_color_streaming(reader, color_mode, debayer_method)?
        }
    };
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    info!(
        selected = selected_indices.len(),
        total, "Selected best frames (color streaming)"
    );
    reporter.finish_stage();

    // Re-read only selected color frames from disk
    reporter.begin_stage(PipelineStage::Reading, Some(selected_indices.len()));
    let selected_color: Vec<ColorFrame> = selected_indices
        .iter()
        .map(|&i| reader.read_frame_as_color(i, debayer_method))
        .collect::<Result<_>>()?;
    let selected_lum: Vec<Frame> = selected_color.iter().map(luminance).collect();
    reporter.finish_stage();

    let stacked_color = if let StackMethod::Drizzle(ref drizzle_config) = config.stacking.method {
        color_drizzle_flow(
            &selected_color,
            &selected_lum,
            &quality_scores,
            backend,
            reporter,
            drizzle_config,
            &config.alignment,
        )?
    } else {
        color_standard_flow(&selected_color, &selected_lum, config, backend, reporter)?
    };

    apply_post_stack_color(stacked_color, config, backend, reporter)
}

fn color_standard_flow(
    selected_color: &[ColorFrame],
    selected_lum: &[Frame],
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<ColorFrame> {
    // Compute alignment offsets on luminance
    let offsets =
        compute_offsets_with_progress(selected_lum, 0, &config.alignment, backend, reporter)?;

    // Apply offsets to each color channel
    let aligned_color = shift_color_frames(selected_color, &offsets);

    // Stack per-channel
    let stack_count = aligned_color.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(stack_count));
    let (red, green, blue) = split_color_channels(&aligned_color);
    let method = &config.stacking.method;
    let result = stack_color_channels_parallel(&red, &green, &blue, method, reporter)?;
    info!(method = ?method, "Color stacking complete");
    reporter.finish_stage();

    Ok(result)
}

fn color_drizzle_flow(
    selected_color: &[ColorFrame],
    selected_lum: &[Frame],
    quality_scores: &[f64],
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &DrizzleConfig,
    alignment_config: &AlignmentConfig,
) -> Result<ColorFrame> {
    // Compute offsets on luminance
    let offsets =
        compute_offsets_with_progress(selected_lum, 0, alignment_config, backend, reporter)?;

    // Drizzle per channel
    let drizzle_count = selected_color.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(drizzle_count));
    let (red, green, blue) = split_color_channels(selected_color);
    let scores = if drizzle_config.quality_weighted && !quality_scores.is_empty() {
        Some(quality_scores)
    } else {
        None
    };
    let result = drizzle_color_channels_parallel(
        &red,
        &green,
        &blue,
        &offsets,
        drizzle_config,
        scores,
        reporter,
    )?;
    info!(
        method = "Drizzle",
        scale = drizzle_config.scale,
        "Color drizzle stacking complete"
    );
    reporter.finish_stage();

    Ok(result)
}

/// Post-stacking processing for color path: sharpen -> filter -> write -> return.
pub(super) fn apply_post_stack_color(
    stacked: ColorFrame,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<PipelineOutput> {
    // Sharpening (per-channel)
    let mut result = if let Some(ref sharpening_config) = config.sharpening {
        reporter.begin_stage(PipelineStage::Sharpening, None);
        let sharpened = process_color_parallel(&stacked, |frame| {
            let mut f = frame.clone();
            if let Some(ref deconv_config) = sharpening_config.deconvolution {
                if backend.is_gpu() {
                    f = deconvolve_gpu(&f, deconv_config, &**backend);
                } else {
                    f = deconvolve(&f, deconv_config);
                }
            }
            wavelet::sharpen(&f, &sharpening_config.wavelet)
        });
        info!("Color sharpening complete");
        reporter.finish_stage();
        sharpened
    } else {
        stacked
    };

    // Filters (per-channel)
    if !config.filters.is_empty() {
        let total_filters = config.filters.len();
        reporter.begin_stage(PipelineStage::Filtering, Some(total_filters));
        for (i, step) in config.filters.iter().enumerate() {
            result = process_color_parallel(&result, |frame| apply_filter_step(frame, step));
            reporter.advance(i + 1);
        }
        info!(count = total_filters, "Color filters applied");
        reporter.finish_stage();
    }

    // Write
    reporter.begin_stage(PipelineStage::Writing, None);
    save_color_image(&result, &config.output)?;
    info!(output = %config.output.display(), "Color output saved");
    reporter.finish_stage();

    Ok(PipelineOutput::Color(result))
}
