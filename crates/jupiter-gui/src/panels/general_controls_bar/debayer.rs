use crate::app::JupiterApp;
use jupiter_core::color::debayer::DebayerMethod;

pub(super) fn debayer_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Debayer", None);
    ui.add_space(4.0);

    if let Some(ref info) = app.ui_state.source_info {
        ui.small(format!("Color mode: {:?}", info.color_mode));
    }

    if ui
        .checkbox(&mut app.config.debayer_enabled, "Enable debayering")
        .changed()
    {
        app.ui_state.mark_dirty_from_score();
    }

    if app.config.debayer_enabled {
        let changed = egui::ComboBox::from_label("Debayer Method")
            .selected_text(app.config.debayer_method.to_string())
            .show_ui(ui, |ui| {
                let mut changed = false;
                for &method in &[DebayerMethod::Bilinear, DebayerMethod::MalvarHeCutler] {
                    if ui
                        .selectable_value(&mut app.config.debayer_method, method, method.to_string())
                        .changed()
                    {
                        changed = true;
                    }
                }
                changed
            });
        if changed.inner == Some(true) {
            app.ui_state.mark_dirty_from_score();
        }
    }
}
