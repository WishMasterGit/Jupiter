use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::states::StackMethodChoice;
use jupiter_core::pipeline::PipelineStage;
use jupiter_core::stack::multi_point::{auto_ap_size, auto_ap_size_from_frame};

pub(super) fn stack_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "3. Stacking",
        app.ui_state.stages.stack.label(),
        app.ui_state.stages.stack.button_color(),
    );
    ui.add_space(4.0);

    let is_multi_point = app.config.stack_method_choice == StackMethodChoice::MultiPoint;
    let enabled = if is_multi_point {
        app.ui_state.stages.score.is_complete()
    } else {
        app.ui_state.stages.align.is_complete()
    };
    ui.add_enabled_ui(enabled, |ui| {
        // Method combo
        let changed = egui::ComboBox::from_label("Method")
            .selected_text(app.config.stack_method_choice.to_string())
            .show_ui(ui, |ui| {
                let mut changed = false;
                for &choice in StackMethodChoice::ALL {
                    if ui
                        .selectable_value(
                            &mut app.config.stack_method_choice,
                            choice,
                            choice.to_string(),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                }
                changed
            });
        if changed.inner == Some(true) {
            app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
        }

        // Method-specific params
        match app.config.stack_method_choice {
            StackMethodChoice::SigmaClip => {
                if ui
                    .add(
                        egui::Slider::new(&mut app.config.sigma_clip_sigma, 0.5..=5.0)
                            .text("Sigma"),
                    )
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
                let mut iter = app.config.sigma_clip_iterations as i32;
                if ui
                    .add(egui::Slider::new(&mut iter, 1..=10).text("Iterations"))
                    .changed()
                {
                    app.config.sigma_clip_iterations = iter as usize;
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
            }
            StackMethodChoice::MultiPoint => {
                // Auto AP size checkbox
                if ui
                    .checkbox(&mut app.config.mp_auto_ap_size, "Auto AP Size")
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }

                if app.config.mp_auto_ap_size {
                    // Compute auto AP size from detected planet or fallback to frame dims
                    let computed = if let Some(diameter) = app.ui_state.detected_planet_diameter {
                        auto_ap_size(diameter)
                    } else if let Some(ref info) = app.ui_state.source_info {
                        auto_ap_size_from_frame(info.width as usize, info.height as usize)
                    } else {
                        app.config.mp_ap_size
                    };
                    if app.config.mp_ap_size != computed {
                        app.config.mp_ap_size = computed;
                        app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                    }
                    ui.label(format!("AP Size: {} px", computed));
                } else {
                    let mut ap = app.config.mp_ap_size as i32;
                    if ui
                        .add(egui::Slider::new(&mut ap, 16..=256).text("AP Size"))
                        .changed()
                    {
                        app.config.mp_ap_size = ap as usize;
                        app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                    }
                }

                let mut sr = app.config.mp_search_radius as i32;
                if ui
                    .add(egui::Slider::new(&mut sr, 4..=64).text("Search Radius"))
                    .changed()
                {
                    app.config.mp_search_radius = sr as usize;
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
                if ui
                    .add(
                        egui::Slider::new(&mut app.config.mp_min_brightness, 0.0..=0.5)
                            .text("Min Bright"),
                    )
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
            }
            StackMethodChoice::Drizzle => {
                if ui
                    .add(
                        egui::Slider::new(&mut app.config.drizzle_scale, 1.0..=4.0).text("Scale"),
                    )
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
                if ui
                    .add(
                        egui::Slider::new(&mut app.config.drizzle_pixfrac, 0.1..=1.0)
                            .text("Pixfrac"),
                    )
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
                if ui
                    .checkbox(&mut app.config.drizzle_quality_weighted, "Quality weighted")
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Stacking);
                }
            }
            _ => {}
        }

        // Stack button
        let can_stack = enabled && !app.ui_state.is_busy();
        if ui
            .add_enabled(can_stack, egui::Button::new("Stack"))
            .clicked()
        {
            app.ui_state
                .stages
                .clear_downstream(PipelineStage::Stacking);
            app.ui_state.running_stage = Some(PipelineStage::Stacking);
            app.send_command(WorkerCommand::Stack {
                method: app.config.stack_method(),
            });
        }
    });
}
