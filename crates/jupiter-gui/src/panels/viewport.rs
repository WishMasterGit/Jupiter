use crate::app::JupiterApp;
use crate::state::{CropRectPixels, crop_aspect_value};

const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 20.0;

pub fn show(ctx: &egui::Context, app: &mut JupiterApp) {
    egui::CentralPanel::default().show(ctx, |ui| {
        // Dark background
        let rect = ui.available_rect_before_wrap();
        ui.painter()
            .rect_filled(rect, 0.0, egui::Color32::from_gray(30));

        // Extract texture info before mutable borrows
        let texture_info = app.viewport.texture.as_ref().map(|t| {
            (t.id(), [t.size()[0] as f32, t.size()[1] as f32])
        });

        if let Some((texture_id, tex_size)) = texture_info {
            // Use original image size for zoom/pan calculations (not texture size which may be scaled)
            let image_size = if let Some(size) = app.viewport.image_size {
                egui::vec2(size[0] as f32, size[1] as f32)
            } else {
                egui::vec2(tex_size[0], tex_size[1])
            };

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

            // Crop drag: primary button without Ctrl, when crop mode active
            let crop_active = app.ui_state.crop_state.active;
            if crop_active {
                handle_crop_drag(&response, ui, app, image_size);

                // Cursor hints for crop interaction
                let cr_scaled = image_size * app.viewport.zoom;
                let cr_center = rect.center() + app.viewport.pan_offset;
                let cr_img_rect = egui::Rect::from_center_size(cr_center, cr_scaled);
                if app.ui_state.crop_state.moving {
                    ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if let Some(hover) = ui.input(|i| i.pointer.hover_pos()) {
                    if response.rect.contains(hover) {
                        let img_pos = screen_to_image(hover, cr_img_rect, image_size);
                        let inside = app.ui_state.crop_state.rect.as_ref().is_some_and(|c| {
                            egui::Rect::from_min_size(
                                egui::pos2(c.x, c.y),
                                egui::vec2(c.width, c.height),
                            )
                            .contains(img_pos)
                        });
                        if inside {
                            ctx.set_cursor_icon(egui::CursorIcon::Grab);
                        } else {
                            ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                        }
                    }
                }
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
                texture_id,
                img_rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Draw crop overlay
            if let Some(ref crop_rect) = app.ui_state.crop_state.rect {
                draw_crop_overlay(ui, crop_rect, img_rect, image_size, app.viewport.zoom);
            }

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

/// Convert screen coordinates to image pixel coordinates.
fn screen_to_image(
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

fn handle_crop_drag(
    response: &egui::Response,
    ui: &egui::Ui,
    app: &mut JupiterApp,
    image_size: egui::Vec2,
) {
    let is_primary = response.dragged_by(egui::PointerButton::Primary);
    let no_ctrl = !ui.input(|i| i.modifiers.command);

    if is_primary && no_ctrl {
        let rect = response.rect;
        let scaled = image_size * app.viewport.zoom;
        let center = rect.center() + app.viewport.pan_offset;
        let img_rect = egui::Rect::from_center_size(center, scaled);

        // Detect drag start: move existing rect or create new selection
        if app.ui_state.crop_state.drag_start.is_none() && !app.ui_state.crop_state.moving {
            if let Some(pos) = response.interact_pointer_pos() {
                let img_pos = screen_to_image(pos, img_rect, image_size);

                let inside = app.ui_state.crop_state.rect.as_ref().is_some_and(|c| {
                    egui::Rect::from_min_size(
                        egui::pos2(c.x, c.y),
                        egui::vec2(c.width, c.height),
                    )
                    .contains(img_pos)
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

        // Move existing crop rect
        if app.ui_state.crop_state.moving {
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
        } else if let Some(start) = app.ui_state.crop_state.drag_start {
            // Create new selection
            if let Some(current) = ui.input(|i| i.pointer.hover_pos()) {
                let img_start = screen_to_image(start, img_rect, image_size);
                let img_current = screen_to_image(current, img_rect, image_size);

                let aspect = crop_aspect_value(app.ui_state.crop_state.aspect_ratio_index);

                if let Some(ratio) = aspect {
                    // Aspect-constrained selection
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

                    // Clamp to image bounds while preserving ratio
                    if w > image_size.x {
                        w = image_size.x;
                        h = w / ratio;
                    }
                    if h > image_size.y {
                        h = image_size.y;
                        w = h * ratio;
                    }

                    // Position based on drag direction
                    let x = if dx >= 0.0 { img_start.x } else { img_start.x - w };
                    let y = if dy >= 0.0 { img_start.y } else { img_start.y - h };

                    let x = x.max(0.0).min(image_size.x - w);
                    let y = y.max(0.0).min(image_size.y - h);

                    if w > 1.0 && h > 1.0 {
                        app.ui_state.crop_state.rect = Some(CropRectPixels {
                            x,
                            y,
                            width: w,
                            height: h,
                        });
                    }
                } else {
                    // Free selection
                    let x_min = img_start.x.min(img_current.x).max(0.0);
                    let y_min = img_start.y.min(img_current.y).max(0.0);
                    let x_max = img_start.x.max(img_current.x).min(image_size.x);
                    let y_max = img_start.y.max(img_current.y).min(image_size.y);

                    let w = x_max - x_min;
                    let h = y_max - y_min;

                    if w > 1.0 && h > 1.0 {
                        app.ui_state.crop_state.rect = Some(CropRectPixels {
                            x: x_min,
                            y: y_min,
                            width: w,
                            height: h,
                        });
                    }
                }
            }
        }
    }

    if response.drag_stopped_by(egui::PointerButton::Primary) {
        app.ui_state.crop_state.drag_start = None;
        app.ui_state.crop_state.moving = false;
        app.ui_state.crop_state.move_offset = None;
    }
}

fn draw_crop_overlay(
    ui: &egui::Ui,
    crop: &CropRectPixels,
    img_rect: egui::Rect,
    image_size: egui::Vec2,
    _zoom: f32,
) {
    let top_left = image_to_screen(
        egui::pos2(crop.x, crop.y),
        img_rect,
        image_size,
    );
    let bottom_right = image_to_screen(
        egui::pos2(crop.x + crop.width, crop.y + crop.height),
        img_rect,
        image_size,
    );
    let crop_screen = egui::Rect::from_min_max(top_left, bottom_right);

    let dim_color = egui::Color32::from_black_alpha(140);
    let painter = ui.painter();

    // Top dim region
    painter.rect_filled(
        egui::Rect::from_min_max(img_rect.left_top(), egui::pos2(img_rect.right(), crop_screen.top())),
        0.0,
        dim_color,
    );
    // Bottom dim region
    painter.rect_filled(
        egui::Rect::from_min_max(egui::pos2(img_rect.left(), crop_screen.bottom()), img_rect.right_bottom()),
        0.0,
        dim_color,
    );
    // Left dim region (between top and bottom)
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(img_rect.left(), crop_screen.top()),
            egui::pos2(crop_screen.left(), crop_screen.bottom()),
        ),
        0.0,
        dim_color,
    );
    // Right dim region (between top and bottom)
    painter.rect_filled(
        egui::Rect::from_min_max(
            egui::pos2(crop_screen.right(), crop_screen.top()),
            egui::pos2(img_rect.right(), crop_screen.bottom()),
        ),
        0.0,
        dim_color,
    );

    // Yellow border around crop rect
    let border_color = egui::Color32::from_rgb(255, 255, 0);
    painter.rect_stroke(crop_screen, 0.0, egui::Stroke::new(1.5, border_color), egui::epaint::StrokeKind::Outside);

    // Dimensions label near bottom-right corner
    let label = format!("{}x{}", crop.width.round() as u32, crop.height.round() as u32);
    let label_pos = egui::pos2(crop_screen.right() - 4.0, crop_screen.bottom() + 4.0);
    painter.text(
        label_pos,
        egui::Align2::RIGHT_TOP,
        label,
        egui::FontId::proportional(12.0),
        border_color,
    );
}

fn fit_to_rect(zoom: &mut f32, pan: &mut egui::Vec2, image_size: egui::Vec2, rect: egui::Rect) {
    let available = rect.size();
    let fit_x = available.x / image_size.x;
    let fit_y = available.y / image_size.y;
    *zoom = fit_x.min(fit_y).clamp(MIN_ZOOM, MAX_ZOOM);
    *pan = egui::Vec2::ZERO;
}
