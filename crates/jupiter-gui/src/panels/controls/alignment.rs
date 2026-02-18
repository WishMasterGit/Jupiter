use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::state::ALIGN_METHOD_NAMES;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn alignment_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "Alignment",
        app.ui_state.align_status.as_deref(),
    );
    ui.add_space(4.0);

    // Keep percentage (frame selection)
    if ui
        .add(
            egui::Slider::new(&mut app.config.select_percentage, 0.01..=1.0)
                .text("Keep %")
                .fixed_decimals(2),
        )
        .changed()
    {
        app.ui_state.mark_dirty_from_align();
    }

    // Method combo
    if egui::ComboBox::from_label("Align Method")
        .selected_text(ALIGN_METHOD_NAMES[app.config.align_method_index])
        .show_index(
            ui,
            &mut app.config.align_method_index,
            ALIGN_METHOD_NAMES.len(),
            |i| ALIGN_METHOD_NAMES[i].to_string(),
        )
        .changed()
    {
        app.ui_state.mark_dirty_from_align();
    }

    // Method-specific params
    match app.config.align_method_index {
        1 => {
            // Enhanced phase correlation
            let mut upsample = app.config.enhanced_phase_upsample as i32;
            if ui
                .add(egui::Slider::new(&mut upsample, 2..=100).text("Upsample"))
                .changed()
            {
                app.config.enhanced_phase_upsample = upsample as usize;
                app.ui_state.mark_dirty_from_align();
            }
        }
        2 => {
            // Centroid
            if ui
                .add(
                    egui::Slider::new(&mut app.config.centroid_threshold, 0.0..=0.5)
                        .text("Threshold"),
                )
                .changed()
            {
                app.ui_state.mark_dirty_from_align();
            }
        }
        4 => {
            // Pyramid
            let mut levels = app.config.pyramid_levels as i32;
            if ui
                .add(egui::Slider::new(&mut levels, 1..=6).text("Levels"))
                .changed()
            {
                app.config.pyramid_levels = levels as usize;
                app.ui_state.mark_dirty_from_align();
            }
        }
        _ => {}
    }

    // Align button
    let can_align = app.ui_state.frames_scored.is_some() && !app.ui_state.is_busy();
    let align_color = if app.ui_state.align_params_dirty {
        Some(egui::Color32::from_rgb(230, 160, 50))
    } else if app.ui_state.align_status.is_some() {
        Some(egui::Color32::from_rgb(80, 180, 80))
    } else {
        None
    };

    let btn = egui::Button::new("Align Frames");
    let btn = if let Some(c) = align_color { btn.fill(c) } else { btn };
    if ui.add_enabled(can_align, btn).clicked() {
        app.ui_state.running_stage = Some(PipelineStage::Alignment);
        app.send_command(WorkerCommand::Align {
            select_percentage: app.config.select_percentage,
            alignment: app.config.alignment_config(),
            device: app.config.device_preference(),
        });
    }
}
