use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use jupiter_core::compute::DevicePreference;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn device_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Compute Device", None, None);
    ui.add_space(4.0);

    egui::ComboBox::from_label("Device")
        .selected_text(app.config.device.to_string())
        .show_ui(ui, |ui| {
            for &pref in &[
                DevicePreference::Auto,
                DevicePreference::Cpu,
                DevicePreference::Gpu,
                DevicePreference::Cuda,
            ] {
                ui.selectable_value(&mut app.config.device, pref, pref.to_string());
            }
        });
}

pub(super) fn actions_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Actions", None, None);
    ui.add_space(4.0);

    // Output path
    ui.horizontal(|ui| {
        ui.label("Output:");
        ui.text_edit_singleline(&mut app.ui_state.output_path);
    });

    ui.add_space(4.0);

    // Run All button
    let can_run = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    if ui
        .add_enabled(can_run, egui::Button::new("Run All").min_size(egui::vec2(ui.available_width(), 28.0)))
        .clicked()
    {
        if let Some(ref path) = app.ui_state.file_path {
            let output = if app.ui_state.output_path.is_empty() {
                path.with_extension("tiff")
            } else {
                std::path::PathBuf::from(&app.ui_state.output_path)
            };
            let config = app.config.to_pipeline_config(path, &output);
            app.ui_state.running_stage = Some(PipelineStage::Reading);
            app.send_command(WorkerCommand::RunAll { config });
        }
    }

    // Save button
    if ui
        .add_enabled(
            !app.ui_state.is_busy(),
            egui::Button::new("Save").min_size(egui::vec2(ui.available_width(), 28.0)),
        )
        .clicked()
    {
        let cmd_tx = app.cmd_tx.clone();
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("TIFF", &["tiff", "tif"])
                .add_filter("PNG", &["png"])
                .set_file_name("output.tiff")
                .save_file()
            {
                let _ = cmd_tx.send(WorkerCommand::SaveImage { path });
            }
        });
    }
}
