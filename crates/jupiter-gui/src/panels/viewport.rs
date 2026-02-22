use crate::app::JupiterApp;
use crate::panels::crop_interaction;

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 20.0;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::CentralPanel::default().show(ctx, |ui| {
        let rect = ui.available_rect_before_wrap();
        paint_background(ui, rect);

        let texture_info = app
            .viewport
            .texture
            .as_ref()
            .map(|t| (t.id(), [t.size()[0] as f32, t.size()[1] as f32]));

        if let Some((texture_id, tex_size)) = texture_info {
            let image_size = resolve_image_size(app, tex_size);
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

            handle_zoom(ui, &response, app, rect);
            handle_pan(ui, &response, app);

            if app.ui_state.crop_state.active {
                crop_interaction::handle_crop_interaction(ctx, &response, ui, app, image_size);
            }

            if response.double_clicked() {
                fit_to_rect(
                    &mut app.viewport.zoom,
                    &mut app.viewport.pan_offset,
                    image_size,
                    rect,
                );
            }

            let img_rect = compute_img_rect(rect, image_size, app);
            draw_image(ui, texture_id, img_rect);

            if let Some(ref crop_rect) = app.ui_state.crop_state.rect {
                crop_interaction::draw_crop_overlay(ui, crop_rect, img_rect, image_size);
            }

            draw_viewing_label(ui, rect, &app.viewport.viewing_label);
        } else {
            show_placeholder(ui);
        }
    });
}

fn paint_background(ui: &egui::Ui, rect: egui::Rect) {
    ui.painter()
        .rect_filled(rect, 0.0, egui::Color32::from_gray(30));
}

fn resolve_image_size(app: &JupiterApp, tex_size: [f32; 2]) -> egui::Vec2 {
    if let Some(size) = app.viewport.image_size {
        egui::vec2(size[0] as f32, size[1] as f32)
    } else {
        egui::vec2(tex_size[0], tex_size[1])
    }
}

fn handle_zoom(ui: &egui::Ui, response: &egui::Response, app: &mut JupiterApp, rect: egui::Rect) {
    let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
    if scroll_delta == 0.0 || !response.hovered() {
        return;
    }

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

fn handle_pan(ui: &egui::Ui, response: &egui::Response, app: &mut JupiterApp) {
    if response.dragged_by(egui::PointerButton::Middle)
        || (response.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.command))
    {
        app.viewport.pan_offset += response.drag_delta();
    }
}

fn compute_img_rect(rect: egui::Rect, image_size: egui::Vec2, app: &JupiterApp) -> egui::Rect {
    let scaled = image_size * app.viewport.zoom;
    let center = rect.center() + app.viewport.pan_offset;
    egui::Rect::from_center_size(center, scaled)
}

fn draw_image(ui: &egui::Ui, texture_id: egui::TextureId, img_rect: egui::Rect) {
    ui.painter().image(
        texture_id,
        img_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

fn draw_viewing_label(ui: &egui::Ui, rect: egui::Rect, label: &str) {
    if label.is_empty() {
        return;
    }
    let label_pos = rect.left_top() + egui::vec2(8.0, 8.0);
    ui.painter().text(
        label_pos,
        egui::Align2::LEFT_TOP,
        label,
        egui::FontId::proportional(14.0),
        egui::Color32::from_white_alpha(200),
    );
}

fn show_placeholder(ui: &mut egui::Ui) {
    ui.centered_and_justified(|ui| {
        ui.label(
            egui::RichText::new("Open video or image to begin")
                .size(18.0)
                .color(egui::Color32::from_gray(100)),
        );
    });
}

fn fit_to_rect(zoom: &mut f32, pan: &mut egui::Vec2, image_size: egui::Vec2, rect: egui::Rect) {
    let available = rect.size();
    let fit_x = available.x / image_size.x;
    let fit_y = available.y / image_size.y;
    *zoom = fit_x.min(fit_y).clamp(MIN_ZOOM, MAX_ZOOM);
    *pan = egui::Vec2::ZERO;
}
