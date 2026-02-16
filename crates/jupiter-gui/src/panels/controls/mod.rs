mod actions;
mod alignment;
mod crop;
mod debayer;
mod file;
mod filters;
mod score;
mod sharpen;
mod stack;

const LEFT_PANEL_WIDTH: f32 = 280.0;

pub fn show(ctx: &egui::Context, app: &mut crate::app::JupiterApp) {
    egui::SidePanel::left("controls")
        .default_width(LEFT_PANEL_WIDTH)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.set_min_width(LEFT_PANEL_WIDTH - 20.0);

                file::file_section(ui, app);
                ui.separator();
                crop::crop_section(ui, app);
                ui.separator();
                debayer::debayer_section(ui, app);
                ui.separator();
                score::score_section(ui, app);
                ui.separator();
                alignment::alignment_section(ui, app);
                ui.separator();
                stack::stack_section(ui, app);
                ui.separator();
                sharpen::sharpen_section(ui, app);
                ui.separator();
                filters::filter_section(ui, app);
                ui.separator();
                actions::device_section(ui, app);
                ui.separator();
                actions::actions_section(ui, app);
            });
        });
}

fn section_header(ui: &mut egui::Ui, label: &str, status: Option<&str>) {
    ui.horizontal(|ui| {
        ui.strong(label);
        if let Some(s) = status {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.small(s);
            });
        }
    });
}
