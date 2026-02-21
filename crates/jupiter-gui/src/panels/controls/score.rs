use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn score_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "1. Frame Selection",
        app.ui_state.stages.score.label(),
        app.ui_state.stages.score.button_color(),
    );
    ui.add_space(4.0);

    // Metric combo
    let changed = egui::ComboBox::from_label("Metric")
        .selected_text(app.config.quality_metric.to_string())
        .show_ui(ui, |ui| {
            let mut changed = false;
            for &metric in &[QualityMetric::Laplacian, QualityMetric::Gradient] {
                if ui
                    .selectable_value(&mut app.config.quality_metric, metric, metric.to_string())
                    .changed()
                {
                    changed = true;
                }
            }
            changed
        });
    if changed.inner == Some(true) {
        app.ui_state.stages.mark_dirty_from(PipelineStage::QualityAssessment);
    }

    // Score button
    let can_score = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    if ui.add_enabled(can_score, egui::Button::new("Score Frames")).clicked() {
        if let Some(path) = app.ui_state.file_path.clone() {
            app.ui_state.stages.clear_downstream(PipelineStage::QualityAssessment);
            app.ui_state.running_stage = Some(PipelineStage::QualityAssessment);
            app.send_command(WorkerCommand::LoadAndScore {
                path,
                metric: app.config.quality_metric,
                debayer: app.config.debayer_config(),
            });
        }
    }
}
