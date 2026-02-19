pub mod config;

use std::sync::Arc;

use tracing::info;

use crate::align::{
    align_frames_configured_with_progress,
    compute_offset_configured,
    compute_offsets_streaming_configured,
    shift_frame,
};
use crate::color::debayer::{debayer, is_bayer, luminance, DebayerMethod};
use crate::color::process::process_color_parallel;
use crate::compute::ComputeBackend;
use crate::error::Result;
use crate::filters::gaussian_blur::gaussian_blur;
use crate::filters::histogram::{auto_stretch, histogram_stretch};
use crate::filters::levels::{brightness_contrast, gamma_correct};
use crate::filters::unsharp_mask::unsharp_mask;
use crate::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame};
use crate::io::image_io::{save_color_image, save_image};
use crate::io::ser::SerReader;
use crate::quality::gradient::{rank_frames_gradient, rank_frames_gradient_color_streaming, rank_frames_gradient_streaming};
use crate::quality::laplacian::{rank_frames, rank_frames_color_streaming, rank_frames_streaming};
use crate::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use crate::sharpen::wavelet;
use crate::stack::drizzle::{drizzle_stack_streaming, drizzle_stack_with_progress};
use crate::stack::mean::{mean_stack_with_progress, StreamingMeanStacker};
use crate::stack::median::median_stack;
use crate::stack::multi_point::{multi_point_stack, multi_point_stack_color};
use crate::stack::sigma_clip::sigma_clip_stack;

use self::config::{FilterStep, MemoryStrategy, PipelineConfig, QualityMetric, StackMethod};

use crate::consts::{COLOR_CHANNEL_COUNT, LOW_MEMORY_THRESHOLD_BYTES};

/// Pipeline processing stage, used for progress reporting.
#[derive(Clone, Copy, Debug)]
pub enum PipelineStage {
    Reading,
    Debayering,
    QualityAssessment,
    FrameSelection,
    Alignment,
    Stacking,
    Sharpening,
    Filtering,
    Writing,
    Cropping,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reading => write!(f, "Reading frames"),
            Self::Debayering => write!(f, "Debayering"),
            Self::QualityAssessment => write!(f, "Assessing quality"),
            Self::FrameSelection => write!(f, "Selecting best frames"),
            Self::Alignment => write!(f, "Aligning frames"),
            Self::Stacking => write!(f, "Stacking"),
            Self::Sharpening => write!(f, "Sharpening"),
            Self::Filtering => write!(f, "Applying filters"),
            Self::Writing => write!(f, "Writing output"),
            Self::Cropping => write!(f, "Cropping"),
        }
    }
}

/// Result of the pipeline — either mono or color.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum PipelineOutput {
    Mono(Frame),
    Color(ColorFrame),
}

impl PipelineOutput {
    /// Get a mono frame. Color output is converted to luminance.
    pub fn to_mono(&self) -> Frame {
        match self {
            Self::Mono(f) => f.clone(),
            Self::Color(cf) => luminance(cf),
        }
    }
}

/// Thread-safe progress reporting for the pipeline.
///
/// Implementors can use this to drive progress bars, logging, or any other
/// UI feedback. All methods have default no-op implementations.
pub trait ProgressReporter: Send + Sync {
    /// A new pipeline stage has started. `total_items` is the number of
    /// work items in this stage (e.g., frame count), if known.
    fn begin_stage(&self, _stage: PipelineStage, _total_items: Option<usize>) {}

    /// One work item within the current stage has completed.
    fn advance(&self, _items_done: usize) {}

    /// The current stage is finished.
    fn finish_stage(&self) {}
}

/// No-op progress reporter, used when `run_pipeline` delegates.
struct NoOpReporter;
impl ProgressReporter for NoOpReporter {}

/// Resolve which debayer method (if any) to use given the config and SER header.
fn resolve_debayer(config: &PipelineConfig, reader: &SerReader) -> Option<DebayerMethod> {
    if config.force_mono {
        return None;
    }
    let mode = reader.header.color_mode();
    if !is_bayer(&mode) && !matches!(mode, ColorMode::RGB | ColorMode::BGR) {
        return None;
    }
    // For RGB/BGR, no debayering needed but we still process as color.
    if matches!(mode, ColorMode::RGB | ColorMode::BGR) {
        // Return a dummy method — it won't be used since we call read_frame_rgb.
        return Some(DebayerMethod::Bilinear);
    }
    // Bayer: use explicit config or auto-detect with default.
    match &config.debayer {
        Some(db) => Some(db.method.clone()),
        None => Some(DebayerMethod::default()),
    }
}

/// Run the full processing pipeline with a thread-safe progress reporter.
pub fn run_pipeline_reported(
    config: &PipelineConfig,
    backend: Arc<dyn ComputeBackend>,
    reporter: Arc<dyn ProgressReporter>,
) -> Result<PipelineOutput> {
    let reader = SerReader::open(&config.input)?;
    let total = reader.frame_count();
    info!(total_frames = total, device = backend.name(), "Reading SER file");

    let debayer_method = resolve_debayer(config, &reader);
    let use_color = debayer_method.is_some();
    let color_mode = reader.header.color_mode();

    if use_color {
        info!(mode = ?color_mode, "Color processing enabled");
    }

    // Multi-point: dedicated flow (color or mono)
    if let StackMethod::MultiPoint(ref mp_config) = config.stacking.method {
        reporter.begin_stage(PipelineStage::Stacking, None);
        if use_color {
            let result = multi_point_stack_color(
                &reader,
                mp_config,
                &color_mode,
                &debayer_method.unwrap(),
                |_progress| {},
            )?;
            info!("Multi-point color stacking complete");
            reporter.finish_stage();
            return apply_post_stack_color(result, config, &backend, &reporter);
        } else {
            let result = multi_point_stack(&reader, mp_config, |_progress| {})?;
            info!("Multi-point stacking complete");
            reporter.finish_stage();
            return apply_post_stack_mono(result, config, &backend, &reporter);
        }
    }

    if use_color {
        run_color_pipeline(&reader, config, &backend, &reporter, &debayer_method.unwrap(), &color_mode, total)
    } else {
        run_mono_pipeline(&reader, config, &backend, &reporter, total)
    }
}

/// Decide whether to use the streaming (low-memory) path.
fn should_use_streaming(reader: &SerReader, config: &PipelineConfig, use_color: bool) -> bool {
    match config.memory {
        MemoryStrategy::Eager => false,
        MemoryStrategy::LowMemory => true,
        MemoryStrategy::Auto => {
            let channels: usize = if use_color { COLOR_CHANNEL_COUNT } else { 1 };
            let frame_bytes = reader.header.width as usize
                * reader.header.height as usize
                * std::mem::size_of::<f32>()
                * channels;
            let total_decoded = frame_bytes * reader.frame_count();
            total_decoded > LOW_MEMORY_THRESHOLD_BYTES
        }
    }
}

/// The existing mono pipeline path (unchanged logic).
fn run_mono_pipeline(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    total: usize,
) -> Result<PipelineOutput> {
    let streaming = should_use_streaming(reader, config, false);
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
    let (selected_indices, _) = select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_frames: Vec<Frame> = selected_indices.iter().map(|&i| frames[i].clone()).collect();
    info!(selected = selected_frames.len(), total, "Selected best frames");
    reporter.finish_stage();

    // Alignment
    let frame_count = selected_frames.len();
    reporter.begin_stage(PipelineStage::Alignment, Some(frame_count));
    let aligned = if frame_count > 1 {
        let r = reporter.clone();
        align_frames_configured_with_progress(&selected_frames, 0, &config.alignment, backend.clone(), move |done| {
            r.advance(done);
        })?
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

/// Streaming mono pipeline: score → select → load-shift-stack one at a time.
///
/// For Mean: fully streaming — each frame is loaded, shifted, accumulated, then dropped.
/// For Median/SigmaClip: semi-streaming — offsets computed streaming, then M selected
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
            let offsets =
                compute_offsets_streaming_configured(reader, &selected_indices, 0, &config.alignment, backend.clone(), move |done| {
                    r.advance(done);
                })?;
            reporter.finish_stage();

            reporter.begin_stage(PipelineStage::Stacking, Some(frame_count));
            let h = reader.header.height as usize;
            let w = reader.header.width as usize;
            let bit_depth = reader.header.pixel_depth as u8;
            let mut stacker = StreamingMeanStacker::new(h, w, bit_depth);
            for (i, (&frame_idx, offset)) in
                selected_indices.iter().zip(offsets.iter()).enumerate()
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
            let offsets =
                compute_offsets_streaming_configured(reader, &selected_indices, 0, &config.alignment, backend.clone(), move |done| {
                    r.advance(done);
                })?;
            reporter.finish_stage();

            // Load and shift selected frames
            reporter.begin_stage(PipelineStage::Reading, Some(frame_count));
            let mut aligned = Vec::with_capacity(frame_count);
            for (i, (&frame_idx, offset)) in
                selected_indices.iter().zip(offsets.iter()).enumerate()
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
            let result = stack_frames_with_progress(&aligned, &config.stacking.method, move |done| {
                r.advance(done);
            })?;
            info!(method = ?config.stacking.method, "Streaming stacking complete");
            reporter.finish_stage();
            Ok(result)
        }
    }
}

/// Streaming mono drizzle: score → select → stream offsets → stream drizzle.
fn run_mono_drizzle_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &crate::stack::drizzle::DrizzleConfig,
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
    let offsets =
        compute_offsets_streaming_configured(reader, &selected_indices, 0, &config.alignment, backend.clone(), move |done| {
            r.advance(done);
        })?;
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
    drizzle_config: &crate::stack::drizzle::DrizzleConfig,
    total: usize,
) -> Result<Frame> {
    // Read
    reporter.begin_stage(PipelineStage::Reading, Some(total));
    let frames: Vec<Frame> = reader.frames().collect::<Result<_>>()?;
    reporter.finish_stage();

    // Quality + selection + offsets + drizzle
    drizzle_flow(&frames, config, backend, reporter, drizzle_config, total)
}

/// Post-stacking processing for mono path: sharpen → filter → write → return.
fn apply_post_stack_mono(
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

// ---------------------------------------------------------------------------
// Color pipeline
// ---------------------------------------------------------------------------

fn run_color_pipeline(
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
            reader, config, backend, reporter, debayer_method, color_mode, total,
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
                debayer(&frame.data, color_mode, debayer_method, frame.original_bit_depth)
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
    info!(selected = selected_color.len(), total, "Selected best frames (color)");
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
        QualityMetric::Laplacian => rank_frames_color_streaming(reader, color_mode, debayer_method)?,
        QualityMetric::Gradient => rank_frames_gradient_color_streaming(reader, color_mode, debayer_method)?,
    };
    reporter.finish_stage();

    // Selection
    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    info!(selected = selected_indices.len(), total, "Selected best frames (color streaming)");
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
    let offsets = compute_offsets_with_progress(
        selected_lum, 0, &config.alignment, backend, reporter,
    )?;

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
    drizzle_config: &crate::stack::drizzle::DrizzleConfig,
    alignment_config: &config::AlignmentConfig,
) -> Result<ColorFrame> {
    // Compute offsets on luminance
    let offsets = compute_offsets_with_progress(
        selected_lum, 0, alignment_config, backend, reporter,
    )?;

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
        &red, &green, &blue, &offsets, drizzle_config, scores, reporter,
    )?;
    info!(
        method = "Drizzle",
        scale = drizzle_config.scale,
        "Color drizzle stacking complete"
    );
    reporter.finish_stage();

    Ok(result)
}

/// Post-stacking processing for color path: sharpen → filter → write → return.
fn apply_post_stack_color(
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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn rank_by_metric(
    frames: &[Frame],
    metric: &QualityMetric,
) -> Vec<(usize, crate::frame::QualityScore)> {
    match metric {
        QualityMetric::Laplacian => rank_frames(frames),
        QualityMetric::Gradient => rank_frames_gradient(frames),
    }
}

/// Streaming variant: score frames one-batch-at-a-time from the SER reader.
fn rank_by_metric_streaming(
    reader: &SerReader,
    metric: &QualityMetric,
) -> Result<Vec<(usize, crate::frame::QualityScore)>> {
    match metric {
        QualityMetric::Laplacian => rank_frames_streaming(reader),
        QualityMetric::Gradient => rank_frames_gradient_streaming(reader),
    }
}

fn select_frames(
    ranked: &[(usize, crate::frame::QualityScore)],
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

fn stack_frames_with_progress(
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
fn compute_offsets_with_progress(
    frames: &[Frame],
    reference_idx: usize,
    alignment_config: &config::AlignmentConfig,
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
fn split_color_channels(frames: &[ColorFrame]) -> (Vec<Frame>, Vec<Frame>, Vec<Frame>) {
    let red: Vec<Frame> = frames.iter().map(|cf| cf.red.clone()).collect();
    let green: Vec<Frame> = frames.iter().map(|cf| cf.green.clone()).collect();
    let blue: Vec<Frame> = frames.iter().map(|cf| cf.blue.clone()).collect();
    (red, green, blue)
}

/// Stack three channel frame lists in parallel using rayon, returning a ColorFrame.
fn stack_color_channels_parallel(
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
fn drizzle_color_channels_parallel(
    red: &[Frame],
    green: &[Frame],
    blue: &[Frame],
    offsets: &[AlignmentOffset],
    drizzle_config: &crate::stack::drizzle::DrizzleConfig,
    scores: Option<&[f64]>,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<ColorFrame> {
    let r = reporter.clone();
    let (dr, (dg, db)) = rayon::join(
        || drizzle_stack_with_progress(red, offsets, drizzle_config, scores, move |done| r.advance(done)),
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
fn shift_color_frames(
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

fn drizzle_flow(
    frames: &[Frame],
    config: &PipelineConfig,
    backend: &Arc<dyn ComputeBackend>,
    reporter: &Arc<dyn ProgressReporter>,
    drizzle_config: &crate::stack::drizzle::DrizzleConfig,
    total: usize,
) -> Result<Frame> {
    reporter.begin_stage(PipelineStage::QualityAssessment, Some(total));
    let ranked = rank_by_metric(frames, &config.frame_selection.metric);
    reporter.finish_stage();

    reporter.begin_stage(PipelineStage::FrameSelection, None);
    let (selected_indices, quality_scores) =
        select_frames(&ranked, total, config.frame_selection.select_percentage);
    let selected_frames: Vec<Frame> = selected_indices.iter().map(|&i| frames[i].clone()).collect();
    info!(
        selected = selected_frames.len(),
        total, "Selected best frames for drizzle"
    );
    reporter.finish_stage();

    // Compute alignment offsets
    let offsets = compute_offsets_with_progress(
        &selected_frames, 0, &config.alignment, backend, reporter,
    )?;
    info!("Alignment offsets computed for drizzle");

    let drizzle_count = selected_frames.len();
    reporter.begin_stage(PipelineStage::Stacking, Some(drizzle_count));
    let scores = if drizzle_config.quality_weighted {
        Some(quality_scores.as_slice())
    } else {
        None
    };
    let r = reporter.clone();
    let result = drizzle_stack_with_progress(&selected_frames, &offsets, drizzle_config, scores, move |done| {
        r.advance(done);
    })?;
    info!(
        method = "Drizzle",
        scale = drizzle_config.scale,
        pixfrac = drizzle_config.pixfrac,
        "Drizzle stacking complete"
    );
    reporter.finish_stage();
    Ok(result)
}

/// Run the full processing pipeline.
///
/// `on_progress` is called with (stage, fraction_complete) for UI updates.
pub fn run_pipeline<F>(
    config: &PipelineConfig,
    backend: Arc<dyn ComputeBackend>,
    _on_progress: F,
) -> Result<PipelineOutput>
where
    F: FnMut(PipelineStage, f32),
{
    let reporter = Arc::new(NoOpReporter);
    run_pipeline_reported(config, backend, reporter)
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
