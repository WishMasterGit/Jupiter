pub mod controls;
pub mod crop_interaction;
pub mod general_controls_bar;
pub mod menu_bar;
pub mod status;
pub mod viewport;

pub(crate) fn section_header(ui: &mut egui::Ui, label: &str, status: Option<&str>) {
    ui.horizontal(|ui| {
        ui.strong(label);
        if let Some(s) = status {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.small(s);
            });
        }
    });
}
