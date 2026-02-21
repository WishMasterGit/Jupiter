pub mod controls;
pub mod crop_interaction;
pub mod general_controls_bar;
pub mod menu_bar;
pub mod status;
pub mod viewport;

pub(crate) fn section_header(
    ui: &mut egui::Ui,
    label: &str,
    status: Option<&str>,
    color: Option<egui::Color32>,
) {
    let frame = if let Some(c) = color {
        egui::Frame::NONE
            .fill(c)
            .inner_margin(4.0)
            .corner_radius(2.0)
    } else {
        egui::Frame::NONE.inner_margin(4.0)
    };
    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.strong(label);
            if let Some(s) = status {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.small(s);
                });
            }
        });
    });
}
