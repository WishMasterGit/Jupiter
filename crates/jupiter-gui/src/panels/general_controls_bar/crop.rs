use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::CropAspect;
use jupiter_core::color::debayer::is_bayer;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn crop_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Crop", None);
    ui.add_space(4.0);

    let file_loaded = app.ui_state.file_path.is_some();
    let busy = app.ui_state.is_busy();
    let enabled = file_loaded && !busy;

    ui.add_enabled_ui(enabled, |ui| {
        ui.toggle_value(&mut app.ui_state.crop_state.active, "Crop");

        if app.ui_state.crop_state.active {
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
                .add_enabled(save_enabled, egui::Button::new("Save Cropped SER..."))
                .clicked()
            {
                let source_path = app.ui_state.file_path.clone().unwrap();
                let crop = core_crop.clone();

                let cmd_tx = app.cmd_tx.clone();
                std::thread::spawn(move || {
                    if let Some(output_path) = rfd::FileDialog::new()
                        .add_filter("SER files", &["ser"])
                        .set_file_name("cropped.ser")
                        .save_file()
                    {
                        let _ = cmd_tx.send(WorkerCommand::CropAndSave {
                            source_path,
                            output_path,
                            crop,
                        });
                    }
                });

                app.ui_state.crop_state.is_saving = true;
                app.ui_state.running_stage = Some(PipelineStage::Cropping);
            }
        });
    }
}
