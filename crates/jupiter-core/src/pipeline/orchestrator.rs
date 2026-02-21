use std::sync::Arc;

use tracing::info;

use crate::color::debayer::{is_bayer, DebayerMethod};
use crate::compute::ComputeBackend;
use crate::consts::{COLOR_CHANNEL_COUNT, LOW_MEMORY_THRESHOLD_BYTES};
use crate::error::Result;
use crate::frame::ColorMode;
use crate::io::ser::SerReader;
use crate::stack::multi_point::{multi_point_stack, multi_point_stack_color};

use super::config::{MemoryStrategy, PipelineConfig, StackMethod};
use super::color::apply_post_stack_color;
use super::mono::apply_post_stack_mono;
use super::types::{NoOpReporter, PipelineOutput, PipelineStage, ProgressReporter};

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
        // Return a dummy method -- it won't be used since we call read_frame_rgb.
        return Some(DebayerMethod::Bilinear);
    }
    // Bayer: use explicit config or auto-detect with default.
    match &config.debayer {
        Some(db) => Some(db.method),
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
    info!(
        total_frames = total,
        device = backend.name(),
        "Reading SER file"
    );

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
        super::color::run_color_pipeline(
            &reader,
            config,
            &backend,
            &reporter,
            &debayer_method.unwrap(),
            &color_mode,
            total,
        )
    } else {
        super::mono::run_mono_pipeline(&reader, config, &backend, &reporter, total)
    }
}

/// Decide whether to use the streaming (low-memory) path.
pub(super) fn should_use_streaming(
    reader: &SerReader,
    config: &PipelineConfig,
    use_color: bool,
) -> bool {
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
