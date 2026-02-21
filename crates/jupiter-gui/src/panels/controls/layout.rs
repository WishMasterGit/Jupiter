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
                    super::score::score_section(ui, app);
                    ui.separator();
                    super::alignment::alignment_section(ui, app);
                    ui.separator();
                    super::stack::stack_section(ui, app);
                    ui.separator();
                }
                super::sharpen::sharpen_section(ui, app);
                ui.separator();
                super::filters::filter_section(ui, app);
            });
        });
}
