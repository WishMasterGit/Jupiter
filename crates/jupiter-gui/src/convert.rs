use jupiter_core::frame::{ColorFrame, Frame};
use jupiter_core::pipeline::PipelineOutput;

/// Maximum texture dimension supported by most GPUs.
const MAX_TEXTURE_SIZE: usize = 8192;

/// Holds a display-ready image along with metadata about scaling.
pub struct DisplayImage {
    pub image: egui::ColorImage,
    pub original_size: [usize; 2],
    pub display_scale: f32,
}

/// Convert a PipelineOutput (mono or color) to a DisplayImage.
pub fn output_to_display_image(output: &PipelineOutput) -> DisplayImage {
    match output {
        PipelineOutput::Mono(frame) => frame_to_display_image(frame),
        PipelineOutput::Color(cf) => color_frame_to_display_image(cf),
    }
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
        let image = frame_to_color_image(frame);
        return DisplayImage {
            image,
            original_size,
            display_scale: 1.0,
        };
    }

    let display_scale = MAX_TEXTURE_SIZE as f32 / max_dim as f32;
    let pixels = bilinear_downscale(original_w, original_h, display_scale, |x0, x1, y0, y1, fx, fy| {
        let v = bilinear_sample_frame(frame, x0, x1, y0, y1, fx, fy);
        let byte = (v.clamp(0.0, 1.0) * 255.0) as u8;
        egui::Color32::from_gray(byte)
    });

    let new_w = ((original_w as f32) * display_scale).round() as usize;
    let new_h = ((original_h as f32) * display_scale).round() as usize;
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

/// Convert a ColorFrame to a DisplayImage, scaling down if it exceeds GPU texture limits.
pub fn color_frame_to_display_image(cf: &ColorFrame) -> DisplayImage {
    let original_h = cf.red.height();
    let original_w = cf.red.width();
    let original_size = [original_w, original_h];

    let max_dim = original_w.max(original_h);
    if max_dim <= MAX_TEXTURE_SIZE {
        let image = color_frame_to_color_image(cf);
        return DisplayImage {
            image,
            original_size,
            display_scale: 1.0,
        };
    }

    let display_scale = MAX_TEXTURE_SIZE as f32 / max_dim as f32;
    let pixels = bilinear_downscale(original_w, original_h, display_scale, |x0, x1, y0, y1, fx, fy| {
        let r = (bilinear_sample_frame(&cf.red, x0, x1, y0, y1, fx, fy).clamp(0.0, 1.0) * 255.0) as u8;
        let g = (bilinear_sample_frame(&cf.green, x0, x1, y0, y1, fx, fy).clamp(0.0, 1.0) * 255.0) as u8;
        let b = (bilinear_sample_frame(&cf.blue, x0, x1, y0, y1, fx, fy).clamp(0.0, 1.0) * 255.0) as u8;
        egui::Color32::from_rgb(r, g, b)
    });

    let new_w = ((original_w as f32) * display_scale).round() as usize;
    let new_h = ((original_h as f32) * display_scale).round() as usize;
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

/// Convert a ColorFrame to an egui ColorImage (RGB).
fn color_frame_to_color_image(cf: &ColorFrame) -> egui::ColorImage {
    let h = cf.red.height();
    let w = cf.red.width();
    let mut pixels = Vec::with_capacity(h * w);

    for row in 0..h {
        for col in 0..w {
            let r = (cf.red.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (cf.green.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (cf.blue.data[[row, col]].clamp(0.0, 1.0) * 255.0) as u8;
            pixels.push(egui::Color32::from_rgb(r, g, b));
        }
    }

    egui::ColorImage {
        size: [w, h],
        pixels,
        source_size: Default::default(),
    }
}

/// Bilinear interpolation of a single sample from a Frame.
fn bilinear_sample_frame(
    frame: &Frame,
    x0: usize, x1: usize,
    y0: usize, y1: usize,
    fx: f32, fy: f32,
) -> f32 {
    frame.data[[y0, x0]] * (1.0 - fx) * (1.0 - fy)
        + frame.data[[y0, x1]] * fx * (1.0 - fy)
        + frame.data[[y1, x0]] * (1.0 - fx) * fy
        + frame.data[[y1, x1]] * fx * fy
}

/// Iterate over a downscaled grid, computing bilinear source coordinates for each
/// destination pixel, and calling `pixel_fn` to produce the output color.
fn bilinear_downscale(
    src_w: usize,
    src_h: usize,
    scale: f32,
    mut pixel_fn: impl FnMut(usize, usize, usize, usize, f32, f32) -> egui::Color32,
) -> Vec<egui::Color32> {
    let new_w = ((src_w as f32) * scale).round() as usize;
    let new_h = ((src_h as f32) * scale).round() as usize;
    let mut pixels = Vec::with_capacity(new_h * new_w);

    for dst_row in 0..new_h {
        for dst_col in 0..new_w {
            let src_x = (dst_col as f32 + 0.5) / scale - 0.5;
            let src_y = (dst_row as f32 + 0.5) / scale - 0.5;

            let x0 = (src_x.floor() as isize).clamp(0, src_w as isize - 1) as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let y0 = (src_y.floor() as isize).clamp(0, src_h as isize - 1) as usize;
            let y1 = (y0 + 1).min(src_h - 1);

            let fx = src_x - src_x.floor();
            let fy = src_y - src_y.floor();

            pixels.push(pixel_fn(x0, x1, y0, y1, fx, fy));
        }
    }

    pixels
}
