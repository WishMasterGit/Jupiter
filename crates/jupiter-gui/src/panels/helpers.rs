/// Show a ComboBox for enum selection. Returns `true` if the value changed.
pub(crate) fn enum_combo<T: PartialEq + Copy + ToString>(
    ui: &mut egui::Ui,
    label: &str,
    current: &mut T,
    options: &[T],
) -> bool {
    let resp = egui::ComboBox::from_label(label)
        .selected_text(current.to_string())
        .show_ui(ui, |ui| {
            let mut changed = false;
            for &choice in options {
                if ui
                    .selectable_value(current, choice, choice.to_string())
                    .changed()
                {
                    changed = true;
                }
            }
            changed
        });
    resp.inner == Some(true)
}

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
