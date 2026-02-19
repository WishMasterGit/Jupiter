mod align;
mod io;
mod pipeline;
mod postprocess;
mod scoring;
mod stacking;

use std::path::PathBuf;
use std::sync::mpsc;

use jupiter_core::frame::{AlignmentOffset, ColorFrame, ColorMode, Frame, QualityScore};
use jupiter_core::pipeline::{PipelineOutput, PipelineStage};

use crate::messages::{WorkerCommand, WorkerResult};

use jupiter_core::color::debayer::DebayerMethod;

/// Cached intermediate results living on the worker thread.
pub(crate) struct PipelineCache {
    pub(crate) file_path: Option<PathBuf>,
    pub(crate) is_color: bool,
    pub(crate) is_streaming: bool,
    /// Stored color mode from the SER header, needed for re-reading color frames in streaming mode.
    pub(crate) color_mode: Option<ColorMode>,
    /// Stored debayer method, needed for re-reading Bayer frames in streaming mode.
    pub(crate) debayer_method: Option<DebayerMethod>,
    pub(crate) all_frames: Option<Vec<Frame>>,
    pub(crate) all_color_frames: Option<Vec<ColorFrame>>,
    pub(crate) ranked: Option<Vec<(usize, QualityScore)>>,
    /// Selected + aligned data (from Align stage).
    pub(crate) selected_frames: Option<Vec<Frame>>,
    pub(crate) selected_color_frames: Option<Vec<ColorFrame>>,
    pub(crate) alignment_offsets: Option<Vec<AlignmentOffset>>,
    /// Quality scores for the selected frames (same order as selected_frames).
    pub(crate) selected_quality_scores: Option<Vec<f64>>,
    /// Result after stacking (mono or color).
    pub(crate) stacked: Option<PipelineOutput>,
    /// Result after sharpening.
    pub(crate) sharpened: Option<PipelineOutput>,
    /// Result after filtering (final output).
    pub(crate) filtered: Option<PipelineOutput>,
}

impl PipelineCache {
    fn new() -> Self {
        Self {
            file_path: None,
            is_color: false,
            is_streaming: false,
            color_mode: None,
            debayer_method: None,
            all_frames: None,
            all_color_frames: None,
            ranked: None,
            selected_frames: None,
            selected_color_frames: None,
            alignment_offsets: None,
            selected_quality_scores: None,
            stacked: None,
            sharpened: None,
            filtered: None,
        }
    }

    /// Latest available output for display/saving.
    pub(crate) fn latest_output(&self) -> Option<PipelineOutput> {
        self.filtered
            .clone()
            .or_else(|| self.sharpened.clone())
            .or_else(|| self.stacked.clone())
    }

    /// Set the stacked result and clear downstream stages.
    pub(crate) fn set_stacked(&mut self, output: PipelineOutput) {
        self.stacked = Some(output);
        self.sharpened = None;
        self.filtered = None;
    }

    pub(crate) fn invalidate_downstream(&mut self) {
        self.selected_frames = None;
        self.selected_color_frames = None;
        self.alignment_offsets = None;
        self.selected_quality_scores = None;
        self.invalidate_from_stack();
    }

    pub(crate) fn invalidate_from_stack(&mut self) {
        self.stacked = None;
        self.sharpened = None;
        self.filtered = None;
    }
}

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
    send(tx, ctx, WorkerResult::Log { message: msg.into() });
}

pub(crate) fn send_error(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    msg: impl Into<String>,
) {
    send(tx, ctx, WorkerResult::Error { message: msg.into() });
}

/// Create a progress callback that sends `WorkerResult::Progress` messages.
pub(crate) fn make_progress_callback(
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
    stage: PipelineStage,
    total: usize,
) -> impl Fn(usize) {
    let tx = tx.clone();
    let ctx = ctx.clone();
    move |done: usize| {
        let _ = tx.send(WorkerResult::Progress {
            stage,
            items_done: Some(done),
            items_total: Some(total),
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
            WorkerCommand::Stack { method } => {
                stacking::handle_stack(&method, &mut cache, &tx, &ctx);
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
        }
    }
}
