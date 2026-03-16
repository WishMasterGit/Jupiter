use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::frame::ColorMode;
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};
use jupiter_core::stack::surface_warp::{
    surface_warp_stack, surface_warp_stack_color, SurfaceWarpConfig,
};

use crate::messages::WorkerResult;

use super::super::{make_normalized_progress_callback, send, send_error, send_log, PipelineCache};

fn surface_warp_detail(frac: f32) -> &'static str {
    match frac {
        f if f <= 0.1 => "Aligning frames",
        f if f <= 0.2 => "Building mean reference",
        f if f <= 0.3 => "Selecting frames",
        f if f < 0.95 => "Warping & accumulating",
        _ => "Finalizing",
    }
}

pub(crate) fn handle_surface_warp(
    sw_config: &SurfaceWarpConfig,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let file_path = match &cache.file_path {
        Some(p) => p.clone(),
        None => {
            send_error(tx, ctx, "No file loaded. Run Score Frames first.");
            return;
        }
    };
    send_log(tx, ctx, "Surface warp stacking...");
    send(
        tx,
        ctx,
        WorkerResult::Progress {
            stage: PipelineStage::Stacking,
            items_done: None,
            items_total: None,
            detail: Some("Starting".into()),
        },
    );
    let start = Instant::now();
    let reader = match SerReader::open(&file_path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open file: {e}"));
            return;
        }
    };

    if cache.is_color {
        let color_mode = match reader.header.color_mode() {
            ColorMode::Mono => {
                send_error(tx, ctx, "Expected color source but got mono");
                return;
            }
            mode => mode,
        };
        let debayer_method = cache.debayer_method.unwrap_or_default();
        let mut on_progress = make_normalized_progress_callback(
            tx,
            ctx,
            PipelineStage::Stacking,
            surface_warp_detail,
        );
        match surface_warp_stack_color(
            &reader,
            sw_config,
            &color_mode,
            &debayer_method,
            &mut on_progress,
        ) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let output = PipelineOutput::Color(result);
                cache.set_stacked(output.clone());
                send_log(
                    tx,
                    ctx,
                    format!(
                        "Surface warp color stacking complete in {:.1}s",
                        elapsed.as_secs_f32()
                    ),
                );
                send(
                    tx,
                    ctx,
                    WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    },
                );
            }
            Err(e) => send_error(tx, ctx, format!("Surface warp color stacking failed: {e}")),
        }
    } else {
        let mut on_progress = make_normalized_progress_callback(
            tx,
            ctx,
            PipelineStage::Stacking,
            surface_warp_detail,
        );
        match surface_warp_stack(&reader, sw_config, &mut on_progress) {
            Ok(result) => {
                let elapsed = start.elapsed();
                let output = PipelineOutput::Mono(result);
                cache.set_stacked(output.clone());
                send_log(
                    tx,
                    ctx,
                    format!(
                        "Surface warp stacking complete in {:.1}s",
                        elapsed.as_secs_f32()
                    ),
                );
                send(
                    tx,
                    ctx,
                    WorkerResult::StackComplete {
                        result: output,
                        elapsed,
                    },
                );
            }
            Err(e) => send_error(tx, ctx, format!("Surface warp stacking failed: {e}")),
        }
    }
}
