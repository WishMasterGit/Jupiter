use crate::align::phase_correlation::{compute_offset, shift_frame};
use crate::error::Result;
use crate::frame::{AlignmentOffset, ColorFrame};

/// Align RGB channels to correct atmospheric dispersion.
///
/// Uses phase correlation between the green channel (reference) and
/// red/blue channels to detect and correct the shift.
pub fn rgb_align(color: &ColorFrame) -> Result<ColorFrame> {
    // Green channel is reference (typically sharpest, middle of spectrum)
    let green_ref = &color.green;

    let red_offset = compute_offset(green_ref, &color.red)?;
    let blue_offset = compute_offset(green_ref, &color.blue)?;

    let aligned_red = shift_frame(&color.red, &red_offset);
    let aligned_blue = shift_frame(&color.blue, &blue_offset);

    Ok(ColorFrame {
        red: aligned_red,
        green: color.green.clone(),
        blue: aligned_blue,
    })
}

/// Align RGB channels with manually specified offsets (in pixels).
pub fn rgb_align_manual(
    color: &ColorFrame,
    red_offset: &AlignmentOffset,
    blue_offset: &AlignmentOffset,
) -> ColorFrame {
    let aligned_red = shift_frame(&color.red, red_offset);
    let aligned_blue = shift_frame(&color.blue, blue_offset);

    ColorFrame {
        red: aligned_red,
        green: color.green.clone(),
        blue: aligned_blue,
    }
}
