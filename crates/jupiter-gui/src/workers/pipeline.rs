use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use jupiter_core::compute::create_backend;
use jupiter_core::pipeline::{run_pipeline_reported, PipelineOutput};

use crate::messages::WorkerResult;
use crate::progress::ChannelProgressReporter;

use super::{send, send_error, send_log, PipelineCache};

pub(super) fn handle_run_all(
    config: &jupiter_core::pipeline::config::PipelineConfig,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, "Running full pipeline...");

    let backend = create_backend(&config.device);
    let reporter = Arc::new(ChannelProgressReporter::new(tx.clone(), ctx.clone()));

    match run_pipeline_reported(config, backend, reporter) {
        Ok(output) => {
            let elapsed = start.elapsed();
            cache.file_path = Some(config.input.clone());
            cache.is_color = matches!(&output, PipelineOutput::Color(_));
            cache.stacked = Some(output.clone());
            cache.sharpened = if config.sharpening.is_some() {
                Some(output.clone())
            } else {
                None
            };
            cache.filtered = if !config.filters.is_empty() {
                Some(output.clone())
            } else {
                None
            };
            send_log(
                tx,
                ctx,
                format!("Pipeline complete in {:.1}s", elapsed.as_secs_f32()),
            );
            send(
                tx,
                ctx,
                WorkerResult::PipelineComplete {
                    result: output,
                    elapsed,
                },
            );
        }
        Err(e) => send_error(tx, ctx, format!("Pipeline failed: {e}")),
    }
}
