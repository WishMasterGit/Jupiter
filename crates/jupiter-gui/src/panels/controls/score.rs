use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::METRIC_NAMES;
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn score_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = app
        .ui_state
        .frames_scored
        .map(|n| format!("{n} scored"));
    super::section_header(
        ui,
        "Frame Selection",
        status.as_deref(),
    );
    ui.add_space(4.0);

    // Metric combo
    let mut metric_idx = match app.config.quality_metric {
        QualityMetric::Laplacian => 0,
        QualityMetric::Gradient => 1,
    };
    let changed_metric = egui::ComboBox::from_label("Metric")
        .selected_text(METRIC_NAMES[metric_idx])
        .show_index(ui, &mut metric_idx, METRIC_NAMES.len(), |i| METRIC_NAMES[i].to_string())
        .changed();
    if changed_metric {
        app.config.quality_metric = match metric_idx {
            0 => QualityMetric::Laplacian,
            _ => QualityMetric::Gradient,
        };
        app.ui_state.score_params_dirty = true;
        app.ui_state.align_params_dirty = true;
        app.ui_state.stack_params_dirty = true;
        app.ui_state.sharpen_params_dirty = true;
        app.ui_state.filter_params_dirty = true;
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
                metric: app.config.quality_metric.clone(),
                debayer: app.config.debayer_config(),
            });
        }
    }
}
