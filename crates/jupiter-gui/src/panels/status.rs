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
        }

        // Log area (last N messages)
        let max_visible = 4;
        let start = app
            .ui_state
            .log_messages
            .len()
            .saturating_sub(max_visible);
        for msg in &app.ui_state.log_messages[start..] {
            ui.small(msg);
        }

        // Status line
        ui.horizontal(|ui| {
            if let Some(ref size) = app.viewport.image_size {
                ui.small(format!("{}x{}", size[0], size[1]));
                ui.separator();
            }
            ui.small(format!("Zoom: {:.0}%", app.viewport.zoom * 100.0));
            ui.separator();
            let device_name = crate::state::DEVICE_NAMES[app.config.device_index];
            ui.small(format!("Device: {device_name}"));
        });

        ui.add_space(2.0);
    });
}
