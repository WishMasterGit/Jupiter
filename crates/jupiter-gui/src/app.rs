use std::sync::mpsc;

use jupiter_core::pipeline::PipelineOutput;

use crate::convert::output_to_display_image;
use crate::messages::{WorkerCommand, WorkerResult};
use crate::panels;
use crate::state::{ConfigState, UIState, ViewportState};
use crate::worker;

pub struct JupiterApp {
    pub cmd_tx: mpsc::Sender<WorkerCommand>,
    pub result_tx: mpsc::Sender<WorkerResult>,
    pub result_rx: mpsc::Receiver<WorkerResult>,
    pub ui_state: UIState,
    pub viewport: ViewportState,
    pub config: ConfigState,
    pub show_about: bool,
}

impl JupiterApp {
    pub fn new(ctx: &egui::Context) -> Self {
        let (result_tx, result_rx) = mpsc::channel();
        let cmd_tx = worker::spawn_worker(result_tx.clone(), ctx.clone());

        Self {
            cmd_tx,
            result_tx,
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
                    self.ui_state.preview_frame_index = 0;
                    self.ui_state.crop_state = Default::default();

                    // Reset pipeline state from previous file
                    self.ui_state.frames_scored = None;
                    self.ui_state.ranked_preview.clear();
                    self.ui_state.align_status = None;
                    self.ui_state.stack_status = None;
                    self.ui_state.sharpen_status = false;
                    self.ui_state.filter_status = None;
                    self.ui_state.clear_all_dirty();
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.viewport.zoom = 1.0;
                    self.viewport.pan_offset = egui::Vec2::ZERO;

                    // Auto-preview frame 0
                    self.send_command(WorkerCommand::PreviewFrame {
                        path: path.clone(),
                        frame_index: 0,
                    });
                    self.ui_state.file_path = Some(path);
                }
                WorkerResult::FramePreview { output, index } => {
                    self.update_viewport_from_output(ctx, &output, &format!("Raw Frame #{index}"));
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
                WorkerResult::AlignComplete { frame_count, elapsed } => {
                    self.ui_state.align_status = Some(format!(
                        "{frame_count} aligned ({})",
                        format_duration(elapsed)
                    ));
                    self.ui_state.running_stage = None;
                    self.ui_state.align_params_dirty = false;
                    self.ui_state.stack_params_dirty = true;
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.ui_state.add_log(format!(
                        "Aligned {frame_count} frames in {}",
                        format_duration(elapsed)
                    ));
                }
                WorkerResult::StackComplete { result, elapsed } => {
                    self.ui_state.stack_status = Some(format!(
                        "Stacked ({})",
                        format_duration(elapsed)
                    ));
                    self.ui_state.running_stage = None;
                    self.ui_state.stack_params_dirty = false;
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.update_viewport_from_output(ctx, &result, "Stacked");
                }
                WorkerResult::SharpenComplete { result, elapsed } => {
                    self.ui_state.sharpen_status = true;
                    self.ui_state.running_stage = None;
                    self.ui_state.sharpen_params_dirty = false;
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.ui_state.add_log(format!("Sharpened in {}", format_duration(elapsed)));
                    self.update_viewport_from_output(ctx, &result, "Sharpened");
                }
                WorkerResult::FilterComplete { result, elapsed } => {
                    self.ui_state.filter_status = Some(self.config.filters.len());
                    self.ui_state.running_stage = None;
                    self.ui_state.filter_params_dirty = false;
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.ui_state.add_log(format!("Filters applied in {}", format_duration(elapsed)));
                    self.update_viewport_from_output(ctx, &result, "Filtered");
                }
                WorkerResult::PipelineComplete { result, elapsed } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.ui_state.add_log(format!(
                        "Pipeline complete in {}",
                        format_duration(elapsed)
                    ));
                    self.update_viewport_from_output(ctx, &result, "Pipeline Result");
                }
                WorkerResult::Progress {
                    stage,
                    items_done,
                    items_total,
                } => {
                    self.ui_state.running_stage = Some(stage);
                    self.ui_state.progress_items_done = items_done;
                    self.ui_state.progress_items_total = items_total;
                }
                WorkerResult::CropComplete { output_path, elapsed } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.crop_state.is_saving = false;
                    self.ui_state.add_log(format!(
                        "Crop saved: {} ({})",
                        output_path.display(),
                        format_duration(elapsed)
                    ));
                }
                WorkerResult::ImageSaved { path } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!("Saved: {}", path.display()));
                }
                WorkerResult::Error { message } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!("ERROR: {message}"));
                }
                WorkerResult::ConfigImported { config } => {
                    self.config = ConfigState::from_pipeline_config(&config);
                    self.ui_state.mark_dirty_from_score();
                    self.ui_state.add_log("Config imported".into());
                }
                WorkerResult::Log { message } => {
                    self.ui_state.add_log(message);
                }
            }
        }
    }

    fn update_viewport_from_output(&mut self, ctx: &egui::Context, output: &PipelineOutput, label: &str) {
        let display = output_to_display_image(output);
        let texture = ctx.load_texture(
            "viewport",
            display.image,
            egui::TextureOptions::NEAREST,
        );
        self.viewport.texture = Some(texture);
        self.viewport.image_size = Some(display.original_size);
        self.viewport.display_scale = display.display_scale;
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
        panels::general_controls_bar::show(ctx, self);
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
