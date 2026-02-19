mod actions;
mod crop;
mod debayer;
mod file;

const RIGHT_PANEL_WIDTH: f32 = 260.0;

pub fn show(ctx: &egui::Context, app: &mut crate::app::JupiterApp) {
    egui::SidePanel::right("general_controls_bar")
        .default_width(RIGHT_PANEL_WIDTH)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.set_min_width(RIGHT_PANEL_WIDTH - 20.0);

                file::file_section(ui, app);
                ui.separator();
                if app.ui_state.is_video {
                    debayer::debayer_section(ui, app);
                    ui.separator();
                }
                crop::crop_section(ui, app);
                ui.separator();
                actions::device_section(ui, app);
                ui.separator();
                actions::actions_section(ui, app);
            });
        });
}
