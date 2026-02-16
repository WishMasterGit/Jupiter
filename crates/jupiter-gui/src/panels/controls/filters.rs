use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::FILTER_TYPE_NAMES;
use jupiter_core::pipeline::config::FilterStep;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn filter_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    let status = app
        .ui_state
        .filter_status
        .map(|n| format!("{n} applied"));
    super::section_header(ui, "Filters", status.as_deref());
    ui.add_space(4.0);

    let mut to_remove = None;
    let mut any_changed = false;
    for (i, filter) in app.config.filters.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("{}.", i + 1));
            match filter {
                FilterStep::AutoStretch {
                    low_percentile,
                    high_percentile,
                } => {
                    ui.label("Auto Stretch");
                    if ui.add(egui::DragValue::new(low_percentile).speed(0.001).prefix("lo: ")).changed() {
                        any_changed = true;
                    }
                    if ui.add(egui::DragValue::new(high_percentile).speed(0.001).prefix("hi: ")).changed() {
                        any_changed = true;
                    }
                }
                FilterStep::HistogramStretch {
                    black_point,
                    white_point,
                } => {
                    ui.label("Hist Stretch");
                    if ui.add(egui::DragValue::new(black_point).speed(0.01).prefix("B: ")).changed() {
                        any_changed = true;
                    }
                    if ui.add(egui::DragValue::new(white_point).speed(0.01).prefix("W: ")).changed() {
                        any_changed = true;
                    }
                }
                FilterStep::Gamma(g) => {
                    ui.label("Gamma");
                    if ui.add(egui::DragValue::new(g).speed(0.05).range(0.1..=5.0)).changed() {
                        any_changed = true;
                    }
                }
                FilterStep::BrightnessContrast {
                    brightness,
                    contrast,
                } => {
                    ui.label("B/C");
                    if ui.add(egui::DragValue::new(brightness).speed(0.01).prefix("B: ")).changed() {
                        any_changed = true;
                    }
                    if ui.add(egui::DragValue::new(contrast).speed(0.05).prefix("C: ")).changed() {
                        any_changed = true;
                    }
                }
                FilterStep::UnsharpMask {
                    radius,
                    amount,
                    threshold,
                } => {
                    ui.label("USM");
                    if ui.add(egui::DragValue::new(radius).speed(0.1).prefix("R: ")).changed() {
                        any_changed = true;
                    }
                    if ui.add(egui::DragValue::new(amount).speed(0.05).prefix("A: ")).changed() {
                        any_changed = true;
                    }
                    if ui.add(egui::DragValue::new(threshold).speed(0.01).prefix("T: ")).changed() {
                        any_changed = true;
                    }
                }
                FilterStep::GaussianBlur { sigma } => {
                    ui.label("Blur");
                    if ui.add(egui::DragValue::new(sigma).speed(0.1).prefix("S: ")).changed() {
                        any_changed = true;
                    }
                }
            }
            if ui.small_button("x").clicked() {
                to_remove = Some(i);
            }
        });
    }

    if any_changed {
        app.ui_state.filter_params_dirty = true;
    }

    if let Some(i) = to_remove {
        app.config.filters.remove(i);
        app.ui_state.filter_params_dirty = true;
    }

    // Add filter menu
    ui.menu_button("+ Add Filter", |ui| {
        for (i, name) in FILTER_TYPE_NAMES.iter().enumerate() {
            if ui.button(*name).clicked() {
                let filter = match i {
                    0 => FilterStep::AutoStretch {
                        low_percentile: 0.001,
                        high_percentile: 0.999,
                    },
                    1 => FilterStep::HistogramStretch {
                        black_point: 0.0,
                        white_point: 1.0,
                    },
                    2 => FilterStep::Gamma(1.0),
                    3 => FilterStep::BrightnessContrast {
                        brightness: 0.0,
                        contrast: 1.0,
                    },
                    4 => FilterStep::UnsharpMask {
                        radius: 1.5,
                        amount: 0.5,
                        threshold: 0.0,
                    },
                    _ => FilterStep::GaussianBlur { sigma: 1.0 },
                };
                app.config.filters.push(filter);
                app.ui_state.filter_params_dirty = true;
                ui.close();
            }
        }
    });

    // Apply button
    let has_base = app.ui_state.stack_status.is_some() || app.ui_state.sharpen_status;
    let can_apply = has_base && !app.ui_state.is_busy() && !app.config.filters.is_empty();
    let filter_color = if app.ui_state.filter_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.filter_status.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Apply Filters");
    let btn = if let Some(c) = filter_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_apply, btn).clicked() {
        app.ui_state.running_stage = Some(PipelineStage::Filtering);
        app.send_command(WorkerCommand::ApplyFilters {
            filters: app.config.filters.clone(),
        });
    }
}
