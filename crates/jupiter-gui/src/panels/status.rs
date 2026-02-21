use crate::app::JupiterApp;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
        ui.add_space(2.0);

        // Progress bar
        if let Some(ref stage) = app.ui_state.running_stage {
            let fraction = match (
                app.ui_state.progress_items_done,
                app.ui_state.progress_items_total,
            ) {
                (Some(done), Some(total)) if total > 0 => done as f32 / total as f32,
                _ => 0.0, // indeterminate
            };

            let detail = match (
                app.ui_state.progress_items_done,
                app.ui_state.progress_items_total,
            ) {
                (Some(done), Some(total)) => format!("{stage} ({done}/{total})"),
                _ => format!("{stage}..."),
            };

            ui.add(egui::ProgressBar::new(fraction).text(detail).animate(true));
        } else {
            // Invisible placeholder — same height, no animation
            ui.add(egui::ProgressBar::new(0.0).text(""));
        }

        // Log area — fixed height for 4 lines, scrollable.
        let line_height = ui.text_style_height(&egui::TextStyle::Body);
        let spacing = ui.spacing().item_spacing.y;
        let log_height = line_height * 4.0 + spacing * 3.0;

        egui::ScrollArea::vertical()
            .max_height(log_height)
            .min_scrolled_height(log_height)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if app.ui_state.log_messages.is_empty() {
                    // Reserve space for 4 empty lines to prevent layout jump.
                    for _ in 0..4 {
                        ui.label("");
                    }
                } else {
                    for msg in &app.ui_state.log_messages {
                        ui.label(msg);
                    }
                }
            });

        // Status line
        ui.horizontal(|ui| {
            if let Some(ref size) = app.viewport.image_size {
                ui.label(format!("{}x{}", size[0], size[1]));
                ui.separator();
            }
            ui.label(format!("Zoom: {:.0}%", app.viewport.zoom * 100.0));
            ui.separator();
            ui.label(format!("Device: {}", app.config.device));
        });

        ui.add_space(2.0);
    });
}
