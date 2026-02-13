use jupiter_core::frame::Frame;

/// Maximum texture dimension supported by most GPUs.
const MAX_TEXTURE_SIZE: usize = 8192;

/// Holds a display-ready image along with metadata about scaling.
pub struct DisplayImage {
    pub image: egui::ColorImage,
    pub original_size: [usize; 2],
    pub display_scale: f32,
}

/// Convert a Frame to a DisplayImage, scaling down if it exceeds GPU texture limits.
/// The original size is preserved for correct zoom/pan calculations.
pub fn frame_to_display_image(frame: &Frame) -> DisplayImage {
    let original_h = frame.height();
    let original_w = frame.width();
    let original_size = [original_w, original_h];

    // Check if scaling is needed
    let max_dim = original_w.max(original_h);
    if max_dim <= MAX_TEXTURE_SIZE {
        // No scaling needed - use the standard conversion
        let image = frame_to_color_image(frame);
        return DisplayImage {
            image,
            original_size,
            display_scale: 1.0,
        };
    }

    // Calculate scale factor to fit within MAX_TEXTURE_SIZE
    let display_scale = MAX_TEXTURE_SIZE as f32 / max_dim as f32;
    let new_w = ((original_w as f32) * display_scale).round() as usize;
    let new_h = ((original_h as f32) * display_scale).round() as usize;

    // Bilinear interpolation to downscale
    let mut pixels = Vec::with_capacity(new_h * new_w);
    for dst_row in 0..new_h {
        for dst_col in 0..new_w {
            // Map destination pixel to source coordinates
            let src_x = (dst_col as f32 + 0.5) / display_scale - 0.5;
            let src_y = (dst_row as f32 + 0.5) / display_scale - 0.5;

            // Bilinear interpolation
            let x0 = (src_x.floor() as isize).clamp(0, original_w as isize - 1) as usize;
            let x1 = (x0 + 1).min(original_w - 1);
            let y0 = (src_y.floor() as isize).clamp(0, original_h as isize - 1) as usize;
            let y1 = (y0 + 1).min(original_h - 1);

            let fx = src_x - src_x.floor();
            let fy = src_y - src_y.floor();

            let v00 = frame.data[[y0, x0]];
            let v10 = frame.data[[y0, x1]];
            let v01 = frame.data[[y1, x0]];
            let v11 = frame.data[[y1, x1]];

            let v = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;

            let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
            pixels.push(egui::Color32::from_gray(byte));
        }
    }

    let image = egui::ColorImage {
        size: [new_w, new_h],
        pixels,
        source_size: Default::default(),
    };

    DisplayImage {
        image,
        original_size,
        display_scale,
    }
}

/// Convert a grayscale Frame (Array2<f32> in [0.0, 1.0]) to an egui ColorImage.
pub fn frame_to_color_image(frame: &Frame) -> egui::ColorImage {
    let h = frame.height();
    let w = frame.width();
    let mut pixels = Vec::with_capacity(h * w);

    for row in 0..h {
        for col in 0..w {
            let v = (frame.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            pixels.push(egui::Color32::from_gray(v));
        }
    }

    egui::ColorImage {
        size: [w, h],
        pixels,
        source_size: Default::default(),
    }
}
