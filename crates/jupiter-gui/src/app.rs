use std::sync::mpsc;

use jupiter_core::frame::Frame;

use crate::convert::frame_to_color_image;
use crate::messages::{WorkerCommand, WorkerResult};
use crate::panels;
use crate::state::{ConfigState, UIState, ViewportState};
use crate::worker;

pub struct JupiterApp {
    pub cmd_tx: mpsc::Sender<WorkerCommand>,
    pub result_rx: mpsc::Receiver<WorkerResult>,
    pub ui_state: UIState,
    pub viewport: ViewportState,
    pub config: ConfigState,
    pub show_about: bool,
}

impl JupiterApp {
    pub fn new(ctx: &egui::Context) -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        let cmd_tx = worker::spawn_worker(result_tx, ctx.clone());

        Self {
            cmd_tx,
            result_rx,
            ui_state: UIState::default(),
            viewport: ViewportState::default(),
            config: ConfigState::default(),
            show_about: false,
        }
    }

    /// Drain all pending results from the worker.
    fn poll_results(&mut self, ctx: &egui::Context) {
        while let Ok(result) = self.result_rx.try_recv() {
            match result {
                WorkerResult::FileInfo { path, info } => {
                    self.ui_state.add_log(format!(
                        "Opened: {} ({}x{}, {} frames, {:?})",
                        path.display(),
                        info.width,
                        info.height,
                        info.total_frames,
                        info.color_mode
                    ));
                    self.ui_state.source_info = Some(info);
                    self.ui_state.file_path = Some(path);
                    self.ui_state.preview_frame_index = 0;
                }
                WorkerResult::FramePreview { frame, index } => {
                    self.update_viewport_texture(ctx, &frame, &format!("Raw Frame #{index}"));
                }
                WorkerResult::LoadAndScoreComplete {
                    frame_count,
                    ranked_preview,
                } => {
                    self.ui_state.frames_scored = Some(frame_count);
                    self.ui_state.ranked_preview = ranked_preview;
                    self.ui_state.running_stage = None;
                    self.ui_state.score_params_dirty = false;
                    self.ui_state.add_log(format!("{frame_count} frames scored"));
                }
                WorkerResult::StackComplete { result, elapsed } => {
                    self.ui_state.stack_status = Some(format!(
                        "Stacked ({})",
                        format_duration(elapsed)
                    ));
                    self.ui_state.running_stage = None;
                    self.ui_state.stack_params_dirty = false;
                    self.update_viewport_texture(ctx, &result, "Stacked");
                }
                WorkerResult::SharpenComplete { result, elapsed } => {
                    self.ui_state.sharpen_status = true;
                    self.ui_state.running_stage = None;
                    self.ui_state.sharpen_params_dirty = false;
                    self.ui_state.add_log(format!("Sharpened in {}", format_duration(elapsed)));
                    self.update_viewport_texture(ctx, &result, "Sharpened");
                }
                WorkerResult::FilterComplete { result, elapsed } => {
                    self.ui_state.filter_status = Some(self.config.filters.len());
                    self.ui_state.running_stage = None;
                    self.ui_state.filter_params_dirty = false;
                    self.ui_state.add_log(format!("Filters applied in {}", format_duration(elapsed)));
                    self.update_viewport_texture(ctx, &result, "Filtered");
                }
                WorkerResult::PipelineComplete { result, elapsed } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!(
                        "Pipeline complete in {}",
                        format_duration(elapsed)
                    ));
                    self.update_viewport_texture(ctx, &result, "Pipeline Result");
                }
                WorkerResult::Progress {
                    stage: _,
                    items_done,
                    items_total,
                } => {
                    self.ui_state.progress_items_done = items_done;
                    self.ui_state.progress_items_total = items_total;
                }
                WorkerResult::ImageSaved { path } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!("Saved: {}", path.display()));
                }
                WorkerResult::Error { message } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!("ERROR: {message}"));
                }
                WorkerResult::Log { message } => {
                    self.ui_state.add_log(message);
                }
            }
        }
    }

    fn update_viewport_texture(&mut self, ctx: &egui::Context, frame: &Frame, label: &str) {
        let image = frame_to_color_image(frame);
        let size = image.size;
        let texture = ctx.load_texture(
            "viewport",
            image,
            egui::TextureOptions::NEAREST,
        );
        self.viewport.texture = Some(texture);
        self.viewport.image_size = Some(size);
        self.viewport.viewing_label = label.to_string();
    }

    pub fn send_command(&self, cmd: WorkerCommand) {
        let _ = self.cmd_tx.send(cmd);
    }
}

impl eframe::App for JupiterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_results(ctx);

        panels::menu_bar::show(ctx, self);
        panels::status::show(ctx, self);
        panels::controls::show(ctx, self);
        panels::viewport::show(ctx, self);

        // About dialog
        if self.show_about {
            egui::Window::new("About Jupiter")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Jupiter");
                        ui.label("Planetary Image Processing");
                        ui.add_space(8.0);
                        ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                        ui.add_space(8.0);
                        if ui.button("Close").clicked() {
                            self.show_about = false;
                        }
                    });
                });
        }
    }
}

fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs_f32();
    if secs < 1.0 {
        format!("{:.0}ms", d.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let mins = secs / 60.0;
        format!("{mins:.1}min")
    }
}
