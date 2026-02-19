use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::color::process::process_color_parallel;
use jupiter_core::compute::create_backend;
use jupiter_core::pipeline::config::{FilterStep, SharpeningConfig};
use jupiter_core::pipeline::{apply_filter_step, PipelineOutput, PipelineStage};
use jupiter_core::sharpen::deconvolution::{deconvolve, deconvolve_gpu};
use jupiter_core::sharpen::wavelet;

use crate::messages::WorkerResult;

use super::{send, send_error, send_log, PipelineCache};

pub(super) fn handle_sharpen(
    config: &SharpeningConfig,
    device: &jupiter_core::compute::DevicePreference,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, "Sharpening...");
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Sharpening,
        items_done: None,
        items_total: None,
    });

    let backend = create_backend(device);

    let sharpen_mono = |frame: &jupiter_core::frame::Frame| -> jupiter_core::frame::Frame {
        let mut result = frame.clone();
        if let Some(ref deconv_config) = config.deconvolution {
            if backend.is_gpu() {
                result = deconvolve_gpu(&result, deconv_config, &*backend);
            } else {
                result = deconvolve(&result, deconv_config);
            }
        }
        wavelet::sharpen(&result, &config.wavelet)
    };

    let stacked = match &cache.stacked {
        Some(s) => s.clone(),
        None => {
            send_error(tx, ctx, "No stacked frame. Run Stack first.");
            return;
        }
    };

    let output = match stacked {
        PipelineOutput::Color(cf) => {
            PipelineOutput::Color(process_color_parallel(&cf, |frame| sharpen_mono(frame)))
        }
        PipelineOutput::Mono(f) => PipelineOutput::Mono(sharpen_mono(&f)),
    };

    let elapsed = start.elapsed();
    cache.sharpened = Some(output.clone());
    cache.filtered = None;
    let label = if cache.is_color {
        "Color sharpening"
    } else {
        "Sharpening"
    };
    send_log(
        tx,
        ctx,
        format!("{label} complete in {:.1}s", elapsed.as_secs_f32()),
    );
    send(tx, ctx, WorkerResult::SharpenComplete {
        result: output,
        elapsed,
    });
}

pub(super) fn handle_apply_filters(
    filters: &[FilterStep],
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, format!("Applying {} filters...", filters.len()));
    send(tx, ctx, WorkerResult::Progress {
        stage: PipelineStage::Filtering,
        items_done: Some(0),
        items_total: Some(filters.len()),
    });

    let base = cache.sharpened.as_ref().or(cache.stacked.as_ref());
    let base = match base {
        Some(b) => b.clone(),
        None => {
            send_error(
                tx,
                ctx,
                "No frame to filter. Run Stack or Sharpen first.",
            );
            return;
        }
    };

    let mut output = base;
    for (i, step) in filters.iter().enumerate() {
        output = match output {
            PipelineOutput::Color(cf) => PipelineOutput::Color(process_color_parallel(&cf, |frame| {
                apply_filter_step(frame, step)
            })),
            PipelineOutput::Mono(f) => PipelineOutput::Mono(apply_filter_step(&f, step)),
        };
        send(tx, ctx, WorkerResult::Progress {
            stage: PipelineStage::Filtering,
            items_done: Some(i + 1),
            items_total: Some(filters.len()),
        });
    }

    let elapsed = start.elapsed();
    cache.filtered = Some(output.clone());
    send_log(
        tx,
        ctx,
        format!(
            "{} filters applied in {:.1}s",
            filters.len(),
            elapsed.as_secs_f32()
        ),
    );
    send(tx, ctx, WorkerResult::FilterComplete {
        result: output,
        elapsed,
    });
}
