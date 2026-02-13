use crate::app::JupiterApp;

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 20.0;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::CentralPanel::default().show(ctx, |ui| {
        // Dark background
        let rect = ui.available_rect_before_wrap();
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(30));

        if let Some(ref texture) = app.viewport.texture {
            let image_size = egui::vec2(
                texture.size()[0] as f32,
                texture.size()[1] as f32,
            );

            // Handle input
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            // Zoom via scroll wheel
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0 && response.hovered() {
                let zoom_factor = (scroll_delta * 0.005).exp();
                let new_zoom = (app.viewport.zoom * zoom_factor).clamp(MIN_ZOOM, MAX_ZOOM);

                // Zoom toward mouse cursor
                if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
                    let center = rect.center().to_vec2() + app.viewport.pan_offset;
                    let mouse_rel = mouse_pos.to_vec2() - center;
                    let scale_change = new_zoom / app.viewport.zoom;
                    app.viewport.pan_offset += mouse_rel * (1.0 - scale_change);
                }

                app.viewport.zoom = new_zoom;
            }

            // Pan via middle-drag or ctrl+left-drag
            if response.dragged_by(egui::PointerButton::Middle)
                || (response.dragged_by(egui::PointerButton::Primary)
                    && ui.input(|i| i.modifiers.command))
            {
                app.viewport.pan_offset += response.drag_delta();
            }

            // Double-click to fit
            if response.double_clicked() {
                fit_to_rect(&mut app.viewport.zoom, &mut app.viewport.pan_offset, image_size, rect);
            }

            // Compute display rect
            let scaled = image_size * app.viewport.zoom;
            let center = rect.center() + app.viewport.pan_offset;
            let img_rect = egui::Rect::from_center_size(center, scaled);

            // Draw the image
            ui.painter().image(
                texture.id(),
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Label in top-left corner
            if !app.viewport.viewing_label.is_empty() {
                let label_pos = rect.left_top() + egui::vec2(8.0, 8.0);
                ui.painter().text(
                    label_pos,
                    egui::Align2::LEFT_TOP,
                    &app.viewport.viewing_label,
                    egui::FontId::proportional(14.0),
                    egui::Color32::from_white_alpha(200),
                );
            }
        } else {
            // No image loaded — show placeholder
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Open a SER file to begin")
                        .size(18.0)
                        .color(egui::Color32::from_gray(100)),
                );
            });
        }
    });
}

fn fit_to_rect(zoom: &mut f32, pan: &mut egui::Vec2, image_size: egui::Vec2, rect: egui::Rect) {
    let available = rect.size();
    let fit_x = available.x / image_size.x;
    let fit_y = available.y / image_size.y;
    *zoom = fit_x.min(fit_y).clamp(MIN_ZOOM, MAX_ZOOM);
    *pan = egui::Vec2::ZERO;
}
