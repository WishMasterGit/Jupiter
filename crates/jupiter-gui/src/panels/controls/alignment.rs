use crate::app::JupiterApp;
use crate::messages::WorkerCommand;
use crate::states::AlignMethodChoice;
use jupiter_core::pipeline::PipelineStage;

pub(super) fn alignment_section(ui: &mut egui::Ui, app: &mut JupiterApp) {
    crate::panels::section_header(
        ui,
        "2. Alignment",
        app.ui_state.stages.align.label(),
        app.ui_state.stages.align.button_color(),
    );
    ui.add_space(4.0);

    let enabled = app.ui_state.stages.score.is_complete();
    ui.add_enabled_ui(enabled, |ui| {
        // Keep percentage (frame selection) — display as percent, store as fraction
        let mut keep_pct = app.config.select_percentage * 100.0;
        if ui
            .add(
                egui::Slider::new(&mut keep_pct, 1.0..=100.0)
                    .text("Keep %")
                    .fixed_decimals(0),
            )
            .changed()
        {
            app.config.select_percentage = keep_pct / 100.0;
            app.ui_state.stages.mark_dirty_from(PipelineStage::Alignment);
        }

        // Method combo
        if crate::panels::enum_combo(ui, "Align Method", &mut app.config.align_method, AlignMethodChoice::ALL) {
            app.ui_state.stages.mark_dirty_from(PipelineStage::Alignment);
        }

        // Method-specific params
        match app.config.align_method {
            AlignMethodChoice::EnhancedPhase => {
                let mut upsample = app.config.enhanced_phase_upsample as i32;
                if ui
                    .add(egui::Slider::new(&mut upsample, 2..=100).text("Upsample"))
                    .changed()
                {
                    app.config.enhanced_phase_upsample = upsample as usize;
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Alignment);
                }
            }
            AlignMethodChoice::Centroid => {
                if ui
                    .add(
                        egui::Slider::new(&mut app.config.centroid_threshold, 0.0..=0.5)
                            .text("Threshold"),
                    )
                    .changed()
                {
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Alignment);
                }
            }
            AlignMethodChoice::Pyramid => {
                let mut levels = app.config.pyramid_levels as i32;
                if ui
                    .add(egui::Slider::new(&mut levels, 1..=6).text("Levels"))
                    .changed()
                {
                    app.config.pyramid_levels = levels as usize;
                    app.ui_state.stages.mark_dirty_from(PipelineStage::Alignment);
                }
            }
            _ => {}
        }

        // Align button
        let can_align = app.ui_state.stages.score.is_complete() && !app.ui_state.is_busy();
        if ui.add_enabled(can_align, egui::Button::new("Align Frames")).clicked() {
            app.ui_state.stages.clear_downstream(PipelineStage::Alignment);
            app.ui_state.running_stage = Some(PipelineStage::Alignment);
            app.send_command(WorkerCommand::Align {
                select_percentage: app.config.select_percentage,
                alignment: app.config.alignment_config(),
                device: app.config.device_preference(),
            });
        }
    });
}
