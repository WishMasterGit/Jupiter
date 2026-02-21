use std::sync::mpsc;

use jupiter_core::pipeline::{PipelineOutput, PipelineStage};

use crate::convert::output_to_display_image;
use crate::messages::{WorkerCommand, WorkerResult};
use crate::panels;
use crate::states::{ConfigState, UIState, ViewportState};
use crate::workers;

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
                    self.ui_state.reset_pipeline();
                    self.viewport.zoom = 1.0;
                    self.viewport.pan_offset = egui::Vec2::ZERO;
                    self.viewport.clear_processed();

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
                    detected_planet_diameter,
                } => {
                    self.ui_state.stages.score.set_complete(format!("{frame_count} scored"));
                    self.ui_state.ranked_preview = ranked_preview;
                    self.ui_state.detected_planet_diameter = detected_planet_diameter;
                    self.ui_state.running_stage = None;
                    self.ui_state.add_log(format!("{frame_count} frames scored"));
                }
                WorkerResult::AlignComplete { frame_count, elapsed } => {
                    self.ui_state.stages.align.set_complete(format!(
                        "{frame_count} aligned ({})",
                        format_duration(elapsed)
                    ));
                    self.ui_state.stages.stack.mark_dirty();
                    self.ui_state.running_stage = None;
                    self.ui_state.clear_progress();
                    self.ui_state.add_log(format!(
                        "Aligned {frame_count} frames in {}",
                        format_duration(elapsed)
                    ));
                }
                WorkerResult::StackComplete { result, elapsed } => {
                    self.ui_state.stages.stack.set_complete(format!(
                        "Stacked ({})",
                        format_duration(elapsed)
                    ));
                    self.ui_state.running_stage = None;
                    self.ui_state.clear_progress();
                    self.ui_state.viewing_raw = false;
                    self.update_processed_output(ctx, &result, "Stacked");
                }
                WorkerResult::SharpenComplete { result, elapsed } => {
                    self.ui_state.stages.sharpen.set_complete("Done".into());
                    self.ui_state.running_stage = None;
                    self.ui_state.clear_progress();
                    self.ui_state.add_log(format!("Sharpened in {}", format_duration(elapsed)));
                    self.update_processed_output(ctx, &result, "Sharpened");
                }
                WorkerResult::FilterComplete { result, elapsed } => {
                    self.ui_state.stages.filter.set_complete(format!("{} applied", self.config.filters.len()));
                    self.ui_state.running_stage = None;
                    self.ui_state.clear_progress();
                    self.ui_state.add_log(format!("Filters applied in {}", format_duration(elapsed)));
                    self.update_processed_output(ctx, &result, "Filtered");
                }
                WorkerResult::PipelineComplete { result, elapsed } => {
                    self.ui_state.running_stage = None;
                    self.ui_state.clear_progress();
                    self.ui_state.add_log(format!(
                        "Pipeline complete in {}",
                        format_duration(elapsed)
                    ));
                    self.ui_state.viewing_raw = false;
                    self.update_processed_output(ctx, &result, "Pipeline Result");
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
                    self.ui_state.reset_pipeline();
                    self.ui_state.stages.stack.set_complete("Image loaded".into());
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
                    self.ui_state.stages.mark_dirty_from(PipelineStage::QualityAssessment);
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

    /// Store a processed result and update viewport only if not viewing raw.
    fn update_processed_output(&mut self, ctx: &egui::Context, output: &PipelineOutput, label: &str) {
        let display = output_to_display_image(output);
        let texture = ctx.load_texture(
            "processed",
            display.image,
            egui::TextureOptions::NEAREST,
        );
        // Always store the processed result for later switching.
        self.viewport.processed_texture = Some(texture.clone());
        self.viewport.processed_image_size = Some(display.original_size);
        self.viewport.processed_display_scale = display.display_scale;
        self.viewport.processed_label = label.to_string();

        // Only update the viewport if viewing processed.
        if !self.ui_state.viewing_raw {
            self.viewport.texture = Some(texture);
            self.viewport.image_size = Some(display.original_size);
            self.viewport.display_scale = display.display_scale;
            self.viewport.viewing_label = label.to_string();
        }
    }

    /// Switch viewport to show the stored processed result.
    pub fn switch_to_processed(&mut self) {
        self.ui_state.viewing_raw = false;
        if let Some(ref tex) = self.viewport.processed_texture {
            self.viewport.texture = Some(tex.clone());
            self.viewport.image_size = self.viewport.processed_image_size;
            self.viewport.display_scale = self.viewport.processed_display_scale;
            self.viewport.viewing_label = self.viewport.processed_label.clone();
        }
    }

    /// Switch viewport to show a raw frame (sends PreviewFrame command).
    pub fn switch_to_raw(&mut self) {
        self.ui_state.viewing_raw = true;
        if let Some(ref path) = self.ui_state.file_path {
            self.send_command(WorkerCommand::PreviewFrame {
                path: path.clone(),
                frame_index: self.ui_state.preview_frame_index,
            });
        }
    }

    pub fn send_command(&self, cmd: WorkerCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// Auto-trigger sharpening when requested (on mouse-up or discrete control change).
    fn check_auto_sharpen(&mut self) {
        if !self.ui_state.sharpen_requested {
            return;
        }
        self.ui_state.sharpen_requested = false;

        let can_sharpen = self.ui_state.stages.stack.is_complete()
            && !self.ui_state.is_busy()
            && self.config.sharpen_enabled
            && self.ui_state.stages.sharpen.is_dirty();

        if can_sharpen {
            if let Some(config) = self.config.sharpening_config() {
                self.ui_state.stages.clear_downstream(PipelineStage::Sharpening);
                self.ui_state.running_stage = Some(PipelineStage::Sharpening);
                self.send_command(WorkerCommand::Sharpen {
                    config,
                    device: self.config.device_preference(),
                });
            }
        }
    }
}

impl eframe::App for JupiterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_results(ctx);
        self.check_auto_sharpen();

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
