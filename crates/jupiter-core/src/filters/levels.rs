use crate::frame::Frame;

/// Apply gamma correction: output = input^(1/gamma).
///
/// gamma > 1.0 brightens midtones, gamma < 1.0 darkens them.
pub fn gamma_correct(frame: &Frame, gamma: f32) -> Frame {
    let inv_gamma = 1.0 / gamma;
    let data = frame.data.mapv(|v| v.clamp(0.0, 1.0).powf(inv_gamma));
    Frame::new(data, frame.original_bit_depth)
}

/// Adjust brightness and contrast.
///
/// `brightness` is added (range roughly -1.0..1.0).
/// `contrast` is multiplied around 0.5 midpoint (1.0 = no change, >1.0 = more contrast).
pub fn brightness_contrast(frame: &Frame, brightness: f32, contrast: f32) -> Frame {
    let data = frame.data.mapv(|v| {
        let adjusted = (v - 0.5) * contrast + 0.5 + brightness;
        adjusted.clamp(0.0, 1.0)
    });
    Frame::new(data, frame.original_bit_depth)
}
