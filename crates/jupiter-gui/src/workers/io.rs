use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use jupiter_core::color::debayer::{is_bayer, DebayerMethod};
use jupiter_core::frame::ColorMode;
use jupiter_core::io::crop::crop_ser;
use jupiter_core::io::image_io::{save_color_image, save_image};
use jupiter_core::io::ser::SerReader;
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};

use crate::messages::WorkerResult;

use super::{send, send_error, send_log, PipelineCache};

pub(super) fn handle_load_file_info(
    path: &Path,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match SerReader::open(path) {
        Ok(reader) => {
            let info = reader.source_info(path);
            send(tx, ctx, WorkerResult::FileInfo {
                path: path.to_path_buf(),
                info,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to open file: {e}")),
    }
}

pub(super) fn handle_preview_frame(
    path: &Path,
    frame_index: usize,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let reader = match SerReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open file: {e}"));
            return;
        }
    };

    let color_mode = reader.header.color_mode();

    let output = if is_bayer(&color_mode) {
        match reader.read_frame_color(frame_index, &DebayerMethod::Bilinear) {
            Ok(cf) => PipelineOutput::Color(cf),
            Err(e) => {
                send_error(
                    tx,
                    ctx,
                    format!("Failed to read frame {frame_index}: {e}"),
                );
                return;
            }
        }
    } else if matches!(color_mode, ColorMode::RGB | ColorMode::BGR) {
        match reader.read_frame_rgb(frame_index) {
            Ok(cf) => PipelineOutput::Color(cf),
            Err(e) => {
                send_error(
                    tx,
                    ctx,
                    format!("Failed to read frame {frame_index}: {e}"),
                );
                return;
            }
        }
    } else {
        match reader.read_frame(frame_index) {
            Ok(frame) => PipelineOutput::Mono(frame),
            Err(e) => {
                send_error(
                    tx,
                    ctx,
                    format!("Failed to read frame {frame_index}: {e}"),
                );
                return;
            }
        }
    };

    send(tx, ctx, WorkerResult::FramePreview {
        output,
        index: frame_index,
    });
}

pub(super) fn handle_save_image(
    path: &Path,
    cache: &PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let output = match cache.latest_output() {
        Some(o) => o,
        None => {
            send_error(tx, ctx, "No frame to save.");
            return;
        }
    };

    let result = match &output {
        PipelineOutput::Mono(frame) => save_image(frame, path),
        PipelineOutput::Color(cf) => save_color_image(cf, path),
    };

    match result {
        Ok(()) => {
            send_log(tx, ctx, format!("Saved to {}", path.display()));
            send(tx, ctx, WorkerResult::ImageSaved {
                path: path.to_path_buf(),
            });
        }
        Err(e) => send_error(tx, ctx, format!("Failed to save: {e}")),
    }
}

pub(super) fn handle_crop_and_save(
    source_path: &Path,
    output_path: &Path,
    crop: &jupiter_core::io::crop::CropRect,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(
        tx,
        ctx,
        format!(
            "Cropping to {}x{} at ({},{})...",
            crop.width, crop.height, crop.x, crop.y
        ),
    );

    let reader = match SerReader::open(source_path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open source: {e}"));
            return;
        }
    };

    let total = reader.frame_count();
    let tx_progress = tx.clone();
    let ctx_progress = ctx.clone();

    match crop_ser(&reader, output_path, crop, |done, total| {
        let _ = tx_progress.send(WorkerResult::Progress {
            stage: PipelineStage::Cropping,
            items_done: Some(done),
            items_total: Some(total),
        });
        ctx_progress.request_repaint();
    }) {
        Ok(()) => {
            let elapsed = start.elapsed();
            send_log(
                tx,
                ctx,
                format!(
                    "Cropped {total} frames in {:.1}s -> {}",
                    elapsed.as_secs_f32(),
                    output_path.display()
                ),
            );
            send(tx, ctx, WorkerResult::CropComplete {
                output_path: output_path.to_path_buf(),
                elapsed,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Crop failed: {e}")),
    }
}
