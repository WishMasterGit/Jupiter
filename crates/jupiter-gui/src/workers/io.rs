use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use std::path::PathBuf;

use jupiter_core::color::debayer::{is_bayer, DebayerMethod};
use jupiter_core::frame::ColorMode;
use jupiter_core::io::autocrop::{auto_detect_crop, AutoCropConfig};
use jupiter_core::io::crop::{crop_ser, CropRect};
use jupiter_core::io::image_io::{
    crop_color_frame, crop_frame, is_color_image, load_color_image, load_image, save_color_image,
    save_image,
};
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

pub(super) fn handle_load_image_file(
    path: &Path,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let is_color = match is_color_image(path) {
        Ok(c) => c,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open image: {e}"));
            return;
        }
    };

    let output = if is_color {
        match load_color_image(path) {
            Ok(cf) => PipelineOutput::Color(cf),
            Err(e) => {
                send_error(tx, ctx, format!("Failed to load image: {e}"));
                return;
            }
        }
    } else {
        match load_image(path) {
            Ok(f) => PipelineOutput::Mono(f),
            Err(e) => {
                send_error(tx, ctx, format!("Failed to load image: {e}"));
                return;
            }
        }
    };

    let (w, h) = match &output {
        PipelineOutput::Mono(f) => (f.width() as u32, f.height() as u32),
        PipelineOutput::Color(cf) => (cf.red.width() as u32, cf.red.height() as u32),
    };

    // Store the loaded image as "stacked" so sharpen/filter stages can work on it.
    *cache = PipelineCache::new();
    cache.file_path = Some(path.to_path_buf());
    cache.is_color = is_color;
    cache.set_stacked(output.clone());

    send(
        tx,
        ctx,
        WorkerResult::ImageLoaded {
            path: path.to_path_buf(),
            output,
            width: w,
            height: h,
        },
    );
}

pub(super) fn handle_crop_and_save_image(
    output_path: &Path,
    crop: &CropRect,
    cache: &PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();

    // Get the original loaded image from the cache (stacked holds it for single images).
    let output = match &cache.stacked {
        Some(o) => o,
        None => {
            send_error(tx, ctx, "No image loaded to crop.");
            return;
        }
    };

    let cropped = match output {
        PipelineOutput::Mono(f) => PipelineOutput::Mono(crop_frame(f, crop)),
        PipelineOutput::Color(cf) => PipelineOutput::Color(crop_color_frame(cf, crop)),
    };

    let result = match &cropped {
        PipelineOutput::Mono(f) => save_image(f, output_path),
        PipelineOutput::Color(cf) => save_color_image(cf, output_path),
    };

    match result {
        Ok(()) => {
            let elapsed = start.elapsed();
            send_log(
                tx,
                ctx,
                format!("Cropped image saved: {}", output_path.display()),
            );
            send(
                tx,
                ctx,
                WorkerResult::CropComplete {
                    output_path: output_path.to_path_buf(),
                    elapsed,
                },
            );
        }
        Err(e) => send_error(tx, ctx, format!("Failed to save cropped image: {e}")),
    }
}

pub(super) fn handle_crop_and_save(
    source_path: &Path,
    output_path: &Path,
    crop: &CropRect,
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

pub(super) fn handle_auto_crop_and_save(
    source_path: &Path,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    let start = Instant::now();
    send_log(tx, ctx, "Auto-detecting planet...");

    let reader = match SerReader::open(source_path) {
        Ok(r) => r,
        Err(e) => {
            send_error(tx, ctx, format!("Failed to open source: {e}"));
            return;
        }
    };

    let config = AutoCropConfig::default();
    let crop = match auto_detect_crop(&reader, &config) {
        Ok(c) => c,
        Err(e) => {
            send_error(tx, ctx, format!("Auto-crop detection failed: {e}"));
            return;
        }
    };

    send_log(
        tx,
        ctx,
        format!(
            "Detected planet: {}x{} at ({},{})",
            crop.width, crop.height, crop.x, crop.y
        ),
    );

    let output_path = auto_crop_output_path(source_path, crop.width, crop.height);
    let total = reader.frame_count();
    let tx_progress = tx.clone();
    let ctx_progress = ctx.clone();

    match crop_ser(&reader, &output_path, &crop, |done, total| {
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
                    "Auto-cropped {total} frames in {:.1}s -> {}",
                    elapsed.as_secs_f32(),
                    output_path.display()
                ),
            );
            send(tx, ctx, WorkerResult::CropComplete {
                output_path,
                elapsed,
            });
        }
        Err(e) => send_error(tx, ctx, format!("Auto-crop failed: {e}")),
    }
}

/// Generate output path: `{stem}_crop{W}x{H}.{ext}`.
fn auto_crop_output_path(source: &Path, crop_w: u32, crop_h: u32) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("ser");
    let parent = source.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}_crop{crop_w}x{crop_h}.{ext}"))
}
