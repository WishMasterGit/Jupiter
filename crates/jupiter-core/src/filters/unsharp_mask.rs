use crate::filters::gaussian_blur::gaussian_blur_array;
use crate::frame::Frame;

/// Apply unsharp mask sharpening.
///
/// `radius` — Gaussian blur sigma for the blurred copy.
/// `amount` — strength of sharpening (e.g. 0.5 = 50% of difference added back).
/// `threshold` — minimum difference to sharpen (prevents sharpening of noise).
pub fn unsharp_mask(frame: &Frame, radius: f32, amount: f32, threshold: f32) -> Frame {
    let blurred = gaussian_blur_array(&frame.data, radius);

    let data = ndarray::Zip::from(&frame.data)
        .and(&blurred)
        .map_collect(|&orig, &blur| {
            let diff = orig - blur;
            if diff.abs() > threshold {
                (orig + diff * amount).clamp(0.0, 1.0)
            } else {
                orig
            }
        });

    Frame::new(data, frame.original_bit_depth)
}
