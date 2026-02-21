use std::path::{Path, PathBuf};

use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::states::CropAspect;
use jupiter_core::color::debayer::is_bayer;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn crop_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Tools", None, None);

    ui.add_space(4.0);

    let file_loaded = app.ui_state.file_path.is_some();
    let busy = app.ui_state.is_busy();
    let enabled = file_loaded && !busy;

    ui.add_enabled_ui(enabled, |ui| {
        ui.toggle_value(&mut app.ui_state.crop_state.active, "Crop");

        if app.ui_state.crop_state.active {
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                app.ui_state.crop_state.active = false;
                return;
            }
            let prev_ratio = app.ui_state.crop_state.aspect_ratio;
            ui.horizontal(|ui| {
                ui.label("Ratio:");
                egui::ComboBox::from_id_salt("crop_aspect_ratio")
                    .selected_text(app.ui_state.crop_state.aspect_ratio.to_string())
                    .width(80.0)
                    .show_ui(ui, |ui| {
                        for &aspect in CropAspect::ALL {
                            ui.selectable_value(
                                &mut app.ui_state.crop_state.aspect_ratio,
                                aspect,
                                aspect.to_string(),
                            );
                        }
                    });
            });

            // Snap existing selection when aspect ratio changes
            if app.ui_state.crop_state.aspect_ratio != prev_ratio {
                if let Some(ratio) = app.ui_state.crop_state.aspect_ratio.ratio() {
                    if let Some(ref mut rect) = app.ui_state.crop_state.rect {
                        let (iw, ih) = app
                            .ui_state
                            .source_info
                            .as_ref()
                            .map(|s| (s.width as f32, s.height as f32))
                            .unwrap_or((1e6, 1e6));
                        rect.snap_to_ratio(ratio, iw, ih);
                    }
                }
            }

            ui.small("Drag to select, drag inside to move");
        } else {
            app.ui_state.crop_state.rect = None
        }
    });

    // Clone rect data out before the mutable closure
    let crop_info = app.ui_state.crop_state.rect.as_ref().map(|r| {
        (
            r.width.round() as u32,
            r.height.round() as u32,
            r.x.round() as u32,
            r.y.round() as u32,
            r.to_core_crop_rect(
                app.ui_state
                    .source_info
                    .as_ref()
                    .map(|i| is_bayer(&i.color_mode))
                    .unwrap_or(false),
            ),
        )
    });
    let is_bayer_mode = app
        .ui_state
        .source_info
        .as_ref()
        .map(|i| is_bayer(&i.color_mode))
        .unwrap_or(false);

    if let Some((w, h, x, y, core_crop)) = crop_info {
        ui.add_space(4.0);
        ui.small(format!("Selection: {w}x{h} at ({x}, {y})"));

        if is_bayer_mode {
            ui.small("(Bayer: will snap to even)");
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(enabled, egui::Button::new("Clear"))
                .clicked()
            {
                app.ui_state.crop_state.rect = None;
            }

            let save_enabled = enabled && !app.ui_state.crop_state.is_saving;
            if ui
                .add_enabled(save_enabled, egui::Button::new("Accept Crop"))
                .clicked()
            {
                let source_path = app.ui_state.file_path.clone().unwrap();
                let output_path = auto_crop_path(&source_path, w, h);
                let crop = core_crop.clone();

                if app.ui_state.is_video {
                    app.send_command(WorkerCommand::CropAndSave {
                        source_path,
                        output_path,
                        crop,
                    });
                } else {
                    app.send_command(WorkerCommand::CropAndSaveImage {
                        output_path,
                        crop,
                    });
                }

                app.ui_state.crop_state.is_saving = true;
                app.ui_state.running_stage = Some(PipelineStage::Cropping);
            }
        });
    }

    // Auto Crop button — only for video (SER) files
    if app.ui_state.is_video {
        ui.add_space(4.0);
        let auto_crop_enabled = enabled && !app.ui_state.crop_state.is_saving;
        if ui
            .add_enabled(auto_crop_enabled, egui::Button::new("Auto Crop"))
            .on_hover_text("Automatically detect the planet and crop")
            .clicked()
        {
            let source_path = app.ui_state.file_path.clone().unwrap();
            app.send_command(WorkerCommand::AutoCropAndSave { source_path });
            app.ui_state.crop_state.is_saving = true;
            app.ui_state.running_stage = Some(PipelineStage::Cropping);
        }
    }
}

/// Generate an output path like `{stem}_crop{W}x{H}.{ext}`.
fn auto_crop_path(source: &Path, crop_w: u32, crop_h: u32) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let ext = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("tiff");
    let parent = source.parent().unwrap_or(Path::new("."));
    parent.join(format!("{stem}_crop{crop_w}x{crop_h}.{ext}"))
}
