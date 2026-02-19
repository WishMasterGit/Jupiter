use std::sync::mpsc;
use std::time::Duration;

use jupiter_core::pipeline::{PipelineOutput, PipelineStage};

use crate::convert::output_to_display_image;
use crate::messages::{WorkerCommand, WorkerResult};
use crate::panels;
use crate::states::{ConfigState, UIState, ViewportState};
use crate::workers;

/// Debounce delay before auto-triggering sharpening after slider changes.
const AUTO_SHARPEN_DEBOUNCE: Duration = Duration::from_millis(300);

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
        let cmd_tx = workers::spawn_worker(result_tx.clone(), ctx.clone());

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
                    self.ui_state.is_video = true;
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
                WorkerResult::ImageLoaded {
                    path,
                    output,
                    width,
                    height,
                } => {
                    self.ui_state.is_video = false;
                    self.ui_state.file_path = Some(path.clone());
                    self.ui_state.source_info = Some(jupiter_core::frame::SourceInfo {
                        filename: path.clone(),
                        total_frames: 1,
                        width,
                        height,
                        bit_depth: 16,
                        color_mode: if matches!(output, PipelineOutput::Color(_)) {
                            jupiter_core::frame::ColorMode::RGB
                        } else {
                            jupiter_core::frame::ColorMode::Mono
                        },
                        observer: None,
                        telescope: None,
                        instrument: None,
                    });
                    self.ui_state.crop_state = Default::default();
                    self.ui_state.frames_scored = None;
                    self.ui_state.ranked_preview.clear();
                    self.ui_state.align_status = None;
                    self.ui_state.stack_status = Some("Image loaded".to_string());
                    self.ui_state.sharpen_status = false;
                    self.ui_state.filter_status = None;
                    self.ui_state.clear_all_dirty();
                    self.ui_state.progress_items_done = None;
                    self.ui_state.progress_items_total = None;
                    self.viewport.zoom = 1.0;
                    self.viewport.pan_offset = egui::Vec2::ZERO;
                    self.ui_state.add_log(format!(
                        "Opened image: {} ({}x{})",
                        path.display(),
                        width,
                        height,
                    ));
                    self.update_viewport_from_output(ctx, &output, "Loaded Image");
                }
                WorkerResult::CropComplete { output_path, elapsed } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.crop_state.is_saving = false;
                    self.ui_state.crop_state.active = false;
                    self.ui_state.crop_state.rect = None;
                    self.ui_state.add_log(format!(
                        "Crop saved: {} ({})",
                        output_path.display(),
                        format_duration(elapsed)
                    ));

                    // Reopen the cropped file
                    let cmd = match output_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.to_ascii_lowercase())
                        .as_deref()
                    {
                        Some("ser") => WorkerCommand::LoadFileInfo {
                            path: output_path,
                        },
                        _ => WorkerCommand::LoadImageFile {
                            path: output_path,
                        },
                    };
                    self.send_command(cmd);
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

    /// Auto-trigger sharpening after a debounce period when sliders change.
    fn check_auto_sharpen(&mut self, ctx: &egui::Context) {
        if let Some(changed_at) = self.ui_state.sharpen_auto_pending {
            let elapsed = changed_at.elapsed();
            if elapsed >= AUTO_SHARPEN_DEBOUNCE {
                let can_sharpen = self.ui_state.stack_status.is_some()
                    && !self.ui_state.is_busy()
                    && self.config.sharpen_enabled
                    && self.ui_state.sharpen_params_dirty;

                if can_sharpen {
                    if let Some(config) = self.config.sharpening_config() {
                        self.ui_state.running_stage = Some(PipelineStage::Sharpening);
                        self.ui_state.sharpen_auto_pending = None;
                        self.send_command(WorkerCommand::Sharpen {
                            config,
                            device: self.config.device_preference(),
                        });
                    }
                } else {
                    self.ui_state.sharpen_auto_pending = None;
                }
            } else {
                ctx.request_repaint_after(AUTO_SHARPEN_DEBOUNCE - elapsed);
            }
        }
    }
}

impl eframe::App for JupiterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_results(ctx);
        self.check_auto_sharpen(ctx);

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
