use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn score_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = app
        .ui_state
        .frames_scored
        .map(|n| format!("{n} scored"));
    crate::panels::section_header(
        ui,
        "Frame Selection",
        status.as_deref(),
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
        app.ui_state.mark_dirty_from_score();
    }

    // Score button
    let can_score = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    let score_color = if app.ui_state.score_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.frames_scored.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Score Frames");
    let btn = if let Some(c) = score_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_score, btn).clicked() {
        if let Some(ref path) = app.ui_state.file_path {
            app.ui_state.running_stage = Some(PipelineStage::QualityAssessment);
            app.send_command(WorkerCommand::LoadAndScore {
                path: path.clone(),
                metric: app.config.quality_metric,
                debayer: app.config.debayer_config(),
            });
        }
    }
}
