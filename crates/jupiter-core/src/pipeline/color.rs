use std::sync::Arc;

use tracing::info;

use crate::align::{
    compute_centering_offsets, compute_common_overlap, detect_centroids, pre_center_color_frames,
    pre_center_frames, shift_frame,
};
use crate::color::debayer::{debayer, luminance, DebayerMethod};
use crate::color::process::process_color_parallel;
use crate::compute::ComputeBackend;
use crate::detection::config::DetectionConfig;
use crate::error::Result;
use crate::frame::{ColorFrame, ColorMode, Frame};
use crate::io::disk_cache::DiskFrameStore;
use crate::io::image_io::save_color_image;
use crate::io::ser::SerReader;
use crate::quality::gradient::rank_frames_gradient_color_streaming;
use crate::quality::laplacian::rank_frames_color_streaming;
use crate::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use crate::sharpen::wavelet;
use crate::stack::disk_backed::{median_stack_disk, sigma_clip_stack_disk, MmapFrameSlice};
use crate::stack::drizzle::drizzle_single_frame;
use crate::stack::sigma_clip::SigmaClipParams;

use super::config::{PipelineConfig, QualityMetric, StackMethod};
use super::helpers::{
    apply_filter_step, compute_offsets_with_progress, drizzle_color_frames, rank_by_metric,
    select_frames, shift_color_frames, split_color_channels, stack_color_channels_parallel,
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
    let (selected_indices, _) =
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

    // Pre-centering (optional, on luminance)
    let (selected_color, selected_lum) = if config.alignment.pre_center {
        apply_pre_centering_color(selected_color, selected_lum, reporter)?
    } else {
        (selected_color, selected_lum)
    };

    let stacked_color =
        color_standard_flow(&selected_color, &selected_lum, config, backend, reporter)?;

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
    let (selected_indices, _) =
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

    // Pre-centering (optional, on luminance)
    let (selected_color, selected_lum) = if config.alignment.pre_center {
        apply_pre_centering_color(selected_color, selected_lum, reporter)?
    } else {
        (selected_color, selected_lum)
    };

    let stacked_color = if matches!(
        config.stacking.method,
        StackMethod::Median | StackMethod::SigmaClip(_)
    ) {
        color_disk_backed_flow(&selected_color, &selected_lum, config, backend, reporter)?
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

    // If drizzle enabled, project each color channel frame onto high-res grid;
    // otherwise shift as before.
    let aligned_color = if let Some(ref drizzle_config) = config.stacking.drizzle {
        info!(
            scale = drizzle_config.scale,
            pixfrac = drizzle_config.pixfrac,
            "Drizzle-projecting color frames"
        );
        drizzle_color_frames(selected_color, &offsets, drizzle_config)?
    } else {
        shift_color_frames(selected_color, &offsets)
    };

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

/// Disk-backed color stacking for median/sigma-clip in streaming mode.
///
/// Computes alignment on luminance, writes shifted per-channel frames to disk,
/// then mmaps them for per-pixel stacking. Peak RAM: one frame at a time during
/// write phase, then OS page-cache paging during stack phase.
fn color_disk_backed_flow(
    selected_color: &[ColorFrame],
    selected_lum: &[Frame],
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<ColorFrame> {
    let offsets =
        compute_offsets_with_progress(selected_lum, 0, &config.alignment, backend, reporter)?;

    let store = if let Some(ref dir) = config.temp_dir {
        DiskFrameStore::with_parent(dir)?
    } else {
        DiskFrameStore::new()?
    };

    let frame_count = selected_color.len();
    let bit_depth = selected_color[0].red.original_bit_depth;
    let drizzle_config = config.stacking.drizzle.as_ref();

    // Write transformed channels to disk
    reporter.begin_stage(PipelineStage::Reading, Some(frame_count));
    let mut disk_red = Vec::with_capacity(frame_count);
    let mut disk_green = Vec::with_capacity(frame_count);
    let mut disk_blue = Vec::with_capacity(frame_count);
    for (i, (cf, offset)) in selected_color.iter().zip(offsets.iter()).enumerate() {
        let (sr, sg, sb) = if let Some(dc) = drizzle_config {
            (
                drizzle_single_frame(&cf.red, offset, dc)?,
                drizzle_single_frame(&cf.green, offset, dc)?,
                drizzle_single_frame(&cf.blue, offset, dc)?,
            )
        } else {
            (
                shift_frame(&cf.red, offset),
                shift_frame(&cf.green, offset),
                shift_frame(&cf.blue, offset),
            )
        };
        disk_red.push(store.store_frame(&sr)?);
        disk_green.push(store.store_frame(&sg)?);
        disk_blue.push(store.store_frame(&sb)?);
        reporter.advance(i + 1);
    }
    reporter.finish_stage();

    // Mmap and stack each channel
    let slices_r: Vec<MmapFrameSlice> = disk_red
        .iter()
        .map(MmapFrameSlice::from_disk_frame)
        .collect::<Result<_>>()?;
    let slices_g: Vec<MmapFrameSlice> = disk_green
        .iter()
        .map(MmapFrameSlice::from_disk_frame)
        .collect::<Result<_>>()?;
    let slices_b: Vec<MmapFrameSlice> = disk_blue
        .iter()
        .map(MmapFrameSlice::from_disk_frame)
        .collect::<Result<_>>()?;

    reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
    let (r, g, b) = match &config.stacking.method {
        StackMethod::Median => (
            median_stack_disk(&slices_r, bit_depth)?,
            median_stack_disk(&slices_g, bit_depth)?,
            median_stack_disk(&slices_b, bit_depth)?,
        ),
        StackMethod::SigmaClip(SigmaClipParams { sigma, iterations }) => (
            sigma_clip_stack_disk(&slices_r, *sigma, *iterations, bit_depth)?,
            sigma_clip_stack_disk(&slices_g, *sigma, *iterations, bit_depth)?,
            sigma_clip_stack_disk(&slices_b, *sigma, *iterations, bit_depth)?,
        ),
        _ => unreachable!(),
    };
    info!(
        method = ?config.stacking.method,
        "Disk-backed color stacking complete"
    );
    reporter.finish_stage();

    Ok(ColorFrame {
        red: r,
        green: g,
        blue: b,
    })
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

/// Apply pre-centering to color frames using luminance-based planet detection.
///
/// Returns updated color frames and luminance frames (both centered + cropped).
fn apply_pre_centering_color(
    selected_color: Vec<ColorFrame>,
    selected_lum: Vec<Frame>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<(Vec<ColorFrame>, Vec<Frame>)> {
    let frame_count = selected_lum.len();
    reporter.begin_stage(PipelineStage::PreCentering, Some(frame_count));
    let detection_config = DetectionConfig::default();
    let centroids = detect_centroids(&selected_lum, &detection_config);
    let (h, w) = (selected_lum[0].height(), selected_lum[0].width());

    let (color_out, lum_out) = if let Some(offsets) = compute_centering_offsets(&centroids, h, w) {
        let crop = compute_common_overlap(&offsets, h, w);
        info!(
            detected = centroids.iter().filter(|c| c.is_some()).count(),
            total = centroids.len(),
            crop = crop.is_some(),
            "Pre-centering color frames"
        );
        let centered_color = pre_center_color_frames(&selected_color, &offsets, crop.as_ref());
        let centered_lum = pre_center_frames(&selected_lum, &offsets, crop.as_ref());
        (centered_color, centered_lum)
    } else {
        info!("Pre-centering skipped: too few planet detections (color)");
        (selected_color, selected_lum)
    };
    reporter.finish_stage();
    Ok((color_out, lum_out))
}
