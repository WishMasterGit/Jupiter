use crate::app::JupiterApp;
use crate::state::CropRectPixels;

/// Convert screen coordinates to image pixel coordinates.
pub fn screen_to_image(
    pos: egui::Pos2,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) -> egui::Pos2 {
    egui::pos2(
        (pos.x - img_rect.left()) / img_rect.width() * image_size.x,
        (pos.y - img_rect.top()) / img_rect.height() * image_size.y,
    )
}

/// Convert image pixel coordinates to screen coordinates.
fn image_to_screen(
    pos: egui::Pos2,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) -> egui::Pos2 {
    egui::pos2(
        pos.x / image_size.x * img_rect.width() + img_rect.left(),
        pos.y / image_size.y * img_rect.height() + img_rect.top(),
    )
}

/// Build an egui::Rect from a CropRectPixels.
fn crop_to_egui_rect(c: &CropRectPixels) -> egui::Rect {
    egui::Rect::from_min_size(
        egui::pos2(c.x, c.y),
        egui::vec2(c.width, c.height),
    )
}

/// Compute the on-screen image rect from viewport state.
fn compute_img_rect(
    panel_rect: egui::Rect,
    image_size: egui::Vec2,
    app: &JupiterApp,
) -> egui::Rect {
    let scaled = image_size * app.viewport.zoom;
    let center = panel_rect.center() + app.viewport.pan_offset;
    egui::Rect::from_center_size(center, scaled)
}

/// Handle crop-related input: drag to create/move selection, and set cursor icons.
pub fn handle_crop_interaction(
    ctx: &egui::Context,
    response: &egui::Response,
    ui: &egui::Ui,
    app: &mut JupiterApp,
    image_size: egui::Vec2,
) {
    handle_crop_drag(response, ui, app, image_size);
    update_crop_cursor(ctx, response, ui, app, image_size);
}

fn handle_crop_drag(
    response: &egui::Response,
    ui: &egui::Ui,
    app: &mut JupiterApp,
    image_size: egui::Vec2,
) {
    let is_primary = response.dragged_by(egui::PointerButton::Primary);
    let no_ctrl = !ui.input(|i| i.modifiers.command);

    if is_primary && no_ctrl {
        let img_rect = compute_img_rect(response.rect, image_size, app);

        if app.ui_state.crop_state.drag_start.is_none() && !app.ui_state.crop_state.moving {
            detect_drag_start(response, app, img_rect, image_size);
        }

        if app.ui_state.crop_state.moving {
            move_existing_crop(ui, app, img_rect, image_size);
        } else if app.ui_state.crop_state.drag_start.is_some() {
            create_new_selection(ui, app, img_rect, image_size);
        }
    }

    if response.drag_stopped_by(egui::PointerButton::Primary) {
        app.ui_state.crop_state.drag_start = None;
        app.ui_state.crop_state.moving = false;
        app.ui_state.crop_state.move_offset = None;
    }
}

fn detect_drag_start(
    response: &egui::Response,
    app: &mut JupiterApp,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) {
    if let Some(pos) = response.interact_pointer_pos() {
        let img_pos = screen_to_image(pos, img_rect, image_size);

        let inside = app.ui_state.crop_state.rect.as_ref().is_some_and(|c| {
            crop_to_egui_rect(c).contains(img_pos)
        });

        if inside {
            let c = app.ui_state.crop_state.rect.as_ref().unwrap();
            app.ui_state.crop_state.moving = true;
            app.ui_state.crop_state.move_offset =
                Some(img_pos.to_vec2() - egui::vec2(c.x, c.y));
        } else {
            app.ui_state.crop_state.drag_start = Some(pos);
        }
    }
}

fn move_existing_crop(
    ui: &egui::Ui,
    app: &mut JupiterApp,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) {
    if let Some(offset) = app.ui_state.crop_state.move_offset {
        if let Some(current) = ui.input(|i| i.pointer.hover_pos()) {
            let img_pos = screen_to_image(current, img_rect, image_size);
            if let Some(ref mut crop) = app.ui_state.crop_state.rect {
                crop.x = (img_pos.x - offset.x)
                    .max(0.0)
                    .min(image_size.x - crop.width);
                crop.y = (img_pos.y - offset.y)
                    .max(0.0)
                    .min(image_size.y - crop.height);
            }
        }
    }
}

fn create_new_selection(
    ui: &egui::Ui,
    app: &mut JupiterApp,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) {
    let start = match app.ui_state.crop_state.drag_start {
        Some(s) => s,
        None => return,
    };

    let current = match ui.input(|i| i.pointer.hover_pos()) {
        Some(c) => c,
        None => return,
    };

    let img_start = screen_to_image(start, img_rect, image_size);
    let img_current = screen_to_image(current, img_rect, image_size);
    let aspect = app.ui_state.crop_state.aspect_ratio.ratio();

    let new_rect = if let Some(ratio) = aspect {
        compute_aspect_constrained_rect(img_start, img_current, ratio, image_size)
    } else {
        compute_free_rect(img_start, img_current, image_size)
    };

    if let Some(r) = new_rect {
        app.ui_state.crop_state.rect = Some(r);
    }
}

fn compute_aspect_constrained_rect(
    img_start: egui::Pos2,
    img_current: egui::Pos2,
    ratio: f32,
    image_size: egui::Vec2,
) -> Option<CropRectPixels> {
    let dx = img_current.x - img_start.x;
    let dy = img_current.y - img_start.y;
    let raw_w = dx.abs();
    let raw_h = dy.abs();

    let h_from_w = raw_w / ratio;
    let (mut w, mut h) = if h_from_w <= raw_h {
        (raw_w, h_from_w)
    } else {
        (raw_h * ratio, raw_h)
    };

    if w > image_size.x {
        w = image_size.x;
        h = w / ratio;
    }
    if h > image_size.y {
        h = image_size.y;
        w = h * ratio;
    }

    let x = if dx >= 0.0 { img_start.x } else { img_start.x - w };
    let y = if dy >= 0.0 { img_start.y } else { img_start.y - h };

    let x = x.max(0.0).min(image_size.x - w);
    let y = y.max(0.0).min(image_size.y - h);

    if w > 1.0 && h > 1.0 {
        Some(CropRectPixels { x, y, width: w, height: h })
    } else {
        None
    }
}

fn compute_free_rect(
    img_start: egui::Pos2,
    img_current: egui::Pos2,
    image_size: egui::Vec2,
) -> Option<CropRectPixels> {
    let x_min = img_start.x.min(img_current.x).max(0.0);
    let y_min = img_start.y.min(img_current.y).max(0.0);
    let x_max = img_start.x.max(img_current.x).min(image_size.x);
    let y_max = img_start.y.max(img_current.y).min(image_size.y);

    let w = x_max - x_min;
    let h = y_max - y_min;

    if w > 1.0 && h > 1.0 {
        Some(CropRectPixels { x: x_min, y: y_min, width: w, height: h })
    } else {
        None
    }
}

fn update_crop_cursor(
    ctx: &egui::Context,
    response: &egui::Response,
    ui: &egui::Ui,
    app: &JupiterApp,
    image_size: egui::Vec2,
) {
    if app.ui_state.crop_state.moving {
        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
        return;
    }

    if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
        if response.rect.contains(hover) {
            let img_rect = compute_img_rect(response.rect, image_size, app);
            let img_pos = screen_to_image(hover, img_rect, image_size);
            let inside = app.ui_state.crop_state.rect.as_ref().is_some_and(|c| {
                crop_to_egui_rect(c).contains(img_pos)
            });
            if inside {
                ctx.set_cursor_icon(egui::CursorIcon::Grab);
            } else {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
            }
        }
    }
}

/// Draw the crop overlay (dim regions + border + dimensions label).
pub fn draw_crop_overlay(
    ui: &egui::Ui,
    crop: &CropRectPixels,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
) {
    let top_left = image_to_screen(egui::pos2(crop.x, crop.y), img_rect, image_size);
    let bottom_right = image_to_screen(
        egui::pos2(crop.x + crop.width, crop.y + crop.height),
        img_rect,
        image_size,
    );
    let crop_screen = egui::Rect::from_min_max(top_left, bottom_right);

    draw_dim_regions(ui, img_rect, crop_screen);
    draw_crop_border(ui, crop_screen);
    draw_dimensions_label(ui, crop, crop_screen);
}

fn draw_dim_regions(ui: &egui::Ui, img_rect: egui::Rect, crop_screen: egui::Rect) {
    let dim_color = egui::Color32::from_black_alpha(140);
    let painter = ui.painter();

    // Top
    painter.rect_filled(
        egui::Rect::from_min_max(img_rect.left_top(), egui::pos2(img_rect.right(), crop_screen.top())),
        0.0,
        dim_color,
    );
    // Bottom
    painter.rect_filled(
        egui::Rect::from_min_max(egui::pos2(img_rect.left(), crop_screen.bottom()), img_rect.right_bottom()),
        0.0,
        dim_color,
    );
    // Left (between top and bottom)
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(img_rect.left(), crop_screen.top()),
            egui::pos2(crop_screen.left(), crop_screen.bottom()),
        ),
        0.0,
        dim_color,
    );
    // Right (between top and bottom)
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(crop_screen.right(), crop_screen.top()),
            egui::pos2(img_rect.right(), crop_screen.bottom()),
        ),
        0.0,
        dim_color,
    );
}

fn draw_crop_border(ui: &egui::Ui, crop_screen: egui::Rect) {
    let border_color = egui::Color32::from_rgb(255, 255, 0);
    ui.painter().rect_stroke(
        crop_screen,
        0.0,
        egui::Stroke::new(1.5, border_color),
        egui::epaint::StrokeKind::Outside,
    );
}

fn draw_dimensions_label(ui: &egui::Ui, crop: &CropRectPixels, crop_screen: egui::Rect) {
    let border_color = egui::Color32::from_rgb(255, 255, 0);
    let label = format!("{}x{}", crop.width.round() as u32, crop.height.round() as u32);
    let label_pos = egui::pos2(crop_screen.right() - 4.0, crop_screen.bottom() + 4.0);
    ui.painter().text(
        label_pos,
        egui::Align2::RIGHT_TOP,
        label,
        egui::FontId::proportional(12.0),
        border_color,
    );
}
