use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::STACK_METHOD_NAMES;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn stack_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "Stacking",
        app.ui_state.stack_status.as_deref(),
    );
    ui.add_space(4.0);

    // Method combo
    if egui::ComboBox::from_label("Method")
        .selected_text(STACK_METHOD_NAMES[app.config.stack_method_index])
        .show_index(ui, &mut app.config.stack_method_index, STACK_METHOD_NAMES.len(), |i| {
            STACK_METHOD_NAMES[i].to_string()
        })
        .changed()
    {
        app.ui_state.mark_dirty_from_stack();
    }

    // Method-specific params
    match app.config.stack_method_index {
        2 => {
            // Sigma clip
            if ui.add(egui::Slider::new(&mut app.config.sigma_clip_sigma, 0.5..=5.0).text("Sigma")).changed() {
                app.ui_state.mark_dirty_from_stack();
            }
            let mut iter = app.config.sigma_clip_iterations as i32;
            if ui.add(egui::Slider::new(&mut iter, 1..=10).text("Iterations")).changed() {
                app.config.sigma_clip_iterations = iter as usize;
                app.ui_state.mark_dirty_from_stack();
            }
        }
        3 => {
            // Multi-point
            let mut ap = app.config.mp_ap_size as i32;
            if ui.add(egui::Slider::new(&mut ap, 16..=256).text("AP Size")).changed() {
                app.config.mp_ap_size = ap as usize;
                app.ui_state.mark_dirty_from_stack();
            }
            let mut sr = app.config.mp_search_radius as i32;
            if ui.add(egui::Slider::new(&mut sr, 4..=64).text("Search Radius")).changed() {
                app.config.mp_search_radius = sr as usize;
                app.ui_state.mark_dirty_from_stack();
            }
            if ui.add(egui::Slider::new(&mut app.config.mp_min_brightness, 0.0..=0.5).text("Min Bright")).changed() {
                app.ui_state.mark_dirty_from_stack();
            }
        }
        4 => {
            // Drizzle
            if ui.add(egui::Slider::new(&mut app.config.drizzle_scale, 1.0..=4.0).text("Scale")).changed() {
                app.ui_state.mark_dirty_from_stack();
            }
            if ui.add(egui::Slider::new(&mut app.config.drizzle_pixfrac, 0.1..=1.0).text("Pixfrac")).changed() {
                app.ui_state.mark_dirty_from_stack();
            }
            if ui.checkbox(&mut app.config.drizzle_quality_weighted, "Quality weighted").changed() {
                app.ui_state.mark_dirty_from_stack();
            }
        }
        _ => {}
    }

    // Stack button
    // Multi-point bypasses alignment stage, only needs scored frames
    let is_multi_point = app.config.stack_method_index == 3;
    let can_stack = if is_multi_point {
        app.ui_state.frames_scored.is_some() && !app.ui_state.is_busy()
    } else {
        app.ui_state.align_status.is_some() && !app.ui_state.is_busy()
    };
    let stack_color = if app.ui_state.stack_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.stack_status.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Stack");
    let btn = if let Some(c) = stack_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_stack, btn).clicked() {
        app.ui_state.running_stage = Some(PipelineStage::Stacking);
        app.send_command(WorkerCommand::Stack {
            method: app.config.stack_method(),
        });
    }
}
