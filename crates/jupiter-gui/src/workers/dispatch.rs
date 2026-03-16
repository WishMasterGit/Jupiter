use std::sync::mpsc;

use jupiter_core::pipeline::PipelineStage;

use crate::messages::{WorkerCommand, WorkerResult};

use super::cache::PipelineCache;
use super::{align, io, pipeline, postprocess, scoring, stacking};

/// Spawn the worker thread. Returns the command sender.
pub fn spawn_worker(
    result_tx: mpsc::Sender<WorkerResult>,
    ctx: egui::Context,
) -> mpsc::Sender<WorkerCommand> {
    let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();

    std::thread::Builder::new()
        .name("jupiter-worker".into())
        .spawn(move || {
            worker_loop(cmd_rx, result_tx, ctx);
        })
        .expect("Failed to spawn worker thread");

    cmd_tx
}

pub(crate) fn send(tx: &mpsc::Sender<WorkerResult>, ctx: &egui::Context, result: WorkerResult) {
    let _ = tx.send(result);
    ctx.request_repaint();
}

pub(crate) fn send_log(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    msg: impl Into<String>,
) {
    send(
        tx,
        ctx,
        WorkerResult::Log {
            message: msg.into(),
        },
    );
}

pub(crate) fn send_error(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    msg: impl Into<String>,
) {
    send(
        tx,
        ctx,
        WorkerResult::Error {
            message: msg.into(),
        },
    );
}

/// Create a progress callback that sends `WorkerResult::Progress` messages.
pub(crate) fn make_progress_callback(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    stage: PipelineStage,
    total: usize,
) -> impl Fn(usize) {
    make_progress_callback_with_detail(tx, ctx, stage, total, None)
}

/// Create a progress callback with a fixed detail string.
pub(crate) fn make_progress_callback_with_detail(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    stage: PipelineStage,
    total: usize,
    detail: Option<String>,
) -> impl Fn(usize) {
    let tx = tx.clone();
    let ctx = ctx.clone();
    move |done: usize| {
        let _ = tx.send(WorkerResult::Progress {
            stage,
            items_done: Some(done),
            items_total: Some(total),
            detail: detail.clone(),
        });
        ctx.request_repaint();
    }
}

/// Create a progress callback for normalized (0.0–1.0) progress from core functions.
///
/// Maps the f32 fraction to 100 discrete steps and uses `detail_fn` to determine
/// the current sub-stage description.
pub(crate) fn make_normalized_progress_callback<D>(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    stage: PipelineStage,
    detail_fn: D,
) -> impl FnMut(f32)
where
    D: Fn(f32) -> &'static str,
{
    let tx = tx.clone();
    let ctx = ctx.clone();
    move |frac: f32| {
        let _ = tx.send(WorkerResult::Progress {
            stage,
            items_done: Some((frac * 100.0) as usize),
            items_total: Some(100),
            detail: Some(detail_fn(frac).into()),
        });
        ctx.request_repaint();
    }
}

fn worker_loop(
    cmd_rx: mpsc::Receiver<WorkerCommand>,
    tx: mpsc::Sender<WorkerResult>,
    ctx: egui::Context,
) {
    let mut cache = PipelineCache::new();

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            WorkerCommand::LoadFileInfo { path } => {
                io::handle_load_file_info(&path, &tx, &ctx);
            }
            WorkerCommand::PreviewFrame { path, frame_index } => {
                io::handle_preview_frame(&path, frame_index, &tx, &ctx);
            }
            WorkerCommand::LoadAndScore {
                path,
                metric,
                debayer,
            } => {
                scoring::handle_load_and_score(&path, &metric, &debayer, &mut cache, &tx, &ctx);
            }
            WorkerCommand::Align {
                select_percentage,
                alignment,
                device,
            } => {
                align::handle_align(
                    select_percentage,
                    &alignment,
                    &device,
                    &mut cache,
                    &tx,
                    &ctx,
                );
            }
            WorkerCommand::Stack { method, drizzle } => {
                stacking::handle_stack(&method, drizzle.as_ref(), &mut cache, &tx, &ctx);
            }
            WorkerCommand::Sharpen { config, device } => {
                postprocess::handle_sharpen(&config, &device, &mut cache, &tx, &ctx);
            }
            WorkerCommand::ApplyFilters { filters } => {
                postprocess::handle_apply_filters(&filters, &mut cache, &tx, &ctx);
            }
            WorkerCommand::RunAll { config } => {
                pipeline::handle_run_all(&config, &mut cache, &tx, &ctx);
            }
            WorkerCommand::SaveImage { path } => {
                io::handle_save_image(&path, &cache, &tx, &ctx);
            }
            WorkerCommand::CropAndSave {
                source_path,
                output_path,
                crop,
            } => {
                io::handle_crop_and_save(&source_path, &output_path, &crop, &tx, &ctx);
            }
            WorkerCommand::LoadImageFile { path } => {
                io::handle_load_image_file(&path, &mut cache, &tx, &ctx);
            }
            WorkerCommand::CropAndSaveImage { output_path, crop } => {
                io::handle_crop_and_save_image(&output_path, &crop, &cache, &tx, &ctx);
            }
            WorkerCommand::AutoCropAndSave { source_path } => {
                io::handle_auto_crop_and_save(&source_path, &tx, &ctx);
            }
        }
    }
}
