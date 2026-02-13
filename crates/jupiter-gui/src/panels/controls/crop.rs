use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use jupiter_core::color::debayer::is_bayer;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn crop_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    super::section_header(ui, "Crop", None);
    ui.add_space(4.0);

    let file_loaded = app.ui_state.file_path.is_some();
    let busy = app.ui_state.is_busy();
    let enabled = file_loaded && !busy;

    ui.add_enabled_ui(enabled, |ui| {
        ui.checkbox(&mut app.ui_state.crop_state.active, "Crop mode");

        if app.ui_state.crop_state.active {
            ui.small("Left-drag on viewport to select region");
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
            if ui.add_enabled(enabled, egui::Button::new("Clear")).clicked() {
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
