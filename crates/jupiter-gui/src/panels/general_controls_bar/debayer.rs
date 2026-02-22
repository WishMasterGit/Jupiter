use crate::app::JupiterApp;
use jupiter_core::color::debayer::DebayerMethod;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn debayer_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(ui, "Debayer", None, None);
    ui.add_space(4.0);

    if let Some(ref info) = app.ui_state.source_info {
        ui.small(format!("Color mode: {:?}", info.color_mode));
    }

    if ui
        .checkbox(&mut app.config.debayer_enabled, "Enable debayering")
        .changed()
    {
        app.ui_state.stages.mark_dirty_from(PipelineStage::QualityAssessment);
    }

    if app.config.debayer_enabled {
        if crate::panels::enum_combo(
            ui,
            "Debayer Method",
            &mut app.config.debayer_method,
            &[DebayerMethod::Bilinear, DebayerMethod::MalvarHeCutler],
        ) {
            app.ui_state.stages.mark_dirty_from(PipelineStage::QualityAssessment);
        }
    }
}
