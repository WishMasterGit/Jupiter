mod alignment;
mod filters;
mod score;
mod sharpen;
mod stack;

const LEFT_PANEL_WIDTH: f32 = 280.0;

pub fn show(ctx: &egui::Context, app: &mut crate::app::JupiterApp) {
    if app.ui_state.file_path.is_none() {
        return;
    }

    egui::SidePanel::left("controls")
        .default_width(LEFT_PANEL_WIDTH)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.set_min_width(LEFT_PANEL_WIDTH - 20.0);

                if app.ui_state.is_video {
                    score::score_section(ui, app);
                    ui.separator();
                    alignment::alignment_section(ui, app);
                    ui.separator();
                    stack::stack_section(ui, app);
                    ui.separator();
                }
                sharpen::sharpen_section(ui, app);
                ui.separator();
                filters::filter_section(ui, app);
            });
        });
}
