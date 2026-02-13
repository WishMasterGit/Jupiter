use jupiter_core::frame::Frame;

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
