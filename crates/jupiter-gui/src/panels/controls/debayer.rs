use crate::app::JupiterApp;
use crate::state::DEBAYER_METHOD_NAMES;

pub(super) fn debayer_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    super::section_header(ui, "Debayer", None);
    ui.add_space(4.0);

    if let Some(ref info) = app.ui_state.source_info {
        ui.small(format!("Color mode: {:?}", info.color_mode));
    }

    if ui
        .checkbox(&mut app.config.debayer_enabled, "Enable debayering")
        .changed()
    {
        app.ui_state.score_params_dirty = true;
    }

    if app.config.debayer_enabled
        && egui::ComboBox::from_label("Debayer Method")
            .selected_text(DEBAYER_METHOD_NAMES[app.config.debayer_method_index])
            .show_index(
                ui,
                &mut app.config.debayer_method_index,
                DEBAYER_METHOD_NAMES.len(),
                |i| DEBAYER_METHOD_NAMES[i].to_string(),
            )
            .changed()
    {
        app.ui_state.score_params_dirty = true;
    }
}
