use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use egui_plot::{Bar, BarChart, HLine, Plot};
use jupiter_core::pipeline::config::QualityMetric;
use jupiter_core::pipeline::PipelineStage;

/// Height of the quality score chart in pixels.
const CHART_HEIGHT: f32 = 120.0;

pub(super) fn score_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "1. Frame Selection",
        app.ui_state.stages.score.label(),
        app.ui_state.stages.score.button_color(),
    );
    ui.add_space(4.0);

    // Metric combo
    if crate::panels::enum_combo(
        ui,
        "Metric",
        &mut app.config.quality_metric,
        &[QualityMetric::Laplacian, QualityMetric::Gradient],
    ) {
        app.ui_state
            .stages
            .mark_dirty_from(PipelineStage::QualityAssessment);
    }

    // Score button
    let can_score = app.ui_state.file_path.is_some() && !app.ui_state.is_busy();
    if ui
        .add_enabled(can_score, egui::Button::new("Score Frames"))
        .clicked()
    {
        if let Some(path) = app.ui_state.file_path.clone() {
            app.ui_state
                .stages
                .clear_downstream(PipelineStage::QualityAssessment);
            app.ui_state.running_stage = Some(PipelineStage::QualityAssessment);
            app.send_command(WorkerCommand::LoadAndScore {
                path,
                metric: app.config.quality_metric,
                debayer: app.config.debayer_config(),
            });
        }
    }

    // Quality score chart
    if !app.ui_state.ranked_preview.is_empty() {
        ui.add_space(4.0);
        quality_chart(
            ui,
            &app.ui_state.ranked_preview,
            app.config.select_percentage,
        );
    }
}

/// Render a bar chart of per-frame quality scores with a keep% cutoff line.
fn quality_chart(ui: &mut egui::Ui, ranked: &[(usize, f64)], keep_percentage: f32) {
    // ranked is sorted by score descending (rank order).
    // Compute the cutoff score from the keep percentage.
    let keep_count = (ranked.len() as f32 * keep_percentage).ceil() as usize;
    let cutoff_score = if keep_count > 0 && keep_count <= ranked.len() {
        ranked[keep_count.saturating_sub(1)].1
    } else if ranked.is_empty() {
        0.0
    } else {
        ranked.last().unwrap().1
    };

    // Build a set of kept frame indices for coloring.
    let kept_indices: std::collections::HashSet<usize> =
        ranked.iter().take(keep_count).map(|(i, _)| *i).collect();

    // Build bars sorted by frame index (original video order).
    let mut by_index: Vec<(usize, f64)> = ranked.to_vec();
    by_index.sort_by_key(|(i, _)| *i);

    // Compute y-axis minimum so bars start near the lowest score, not 0.
    let min_score = ranked.iter().map(|(_, s)| *s).fold(f64::INFINITY, f64::min);
    let max_score = ranked
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = (max_score - min_score) * 0.05;
    let y_min = (min_score - margin).max(0.0);

    let kept_color = egui::Color32::from_rgb(80, 180, 80);
    let rejected_color = egui::Color32::from_rgb(128, 128, 128);

    let bars: Vec<Bar> = by_index
        .iter()
        .map(|(frame_idx, score)| {
            let color = if kept_indices.contains(frame_idx) {
                kept_color
            } else {
                rejected_color
            };
            Bar::new(*frame_idx as f64, score - y_min)
                .fill(color)
                .width(0.8)
                .base_offset(y_min)
        })
        .collect();

    let chart = BarChart::new("quality", bars);

    let cutoff_line = HLine::new("keep cutoff", cutoff_score)
        .color(egui::Color32::from_rgb(255, 160, 40))
        .width(1.5);

    Plot::new("quality_score_chart")
        .height(CHART_HEIGHT)
        .include_y(y_min)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_boxed_zoom(false)
        .show_grid(false)
        .y_axis_label("score")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(chart);
            plot_ui.hline(cutoff_line);
        });
}
