use ndarray::Array2;

use crate::frame::{ColorFrame, Frame};

/// Split an interleaved RGB Array2 (shape: height x width*3) into separate R, G, B frames.
///
/// Assumes the input has 3 values per pixel packed as [R, G, B, R, G, B, ...] across columns.
pub fn split_rgb(data: &Array2<f32>, bit_depth: u8) -> ColorFrame {
    let (h, w3) = data.dim();
    let w = w3 / 3;

    let mut red = Array2::<f32>::zeros((h, w));
    let mut green = Array2::<f32>::zeros((h, w));
    let mut blue = Array2::<f32>::zeros((h, w));

    for row in 0..h {
        for col in 0..w {
            red[[row, col]] = data[[row, col * 3]];
            green[[row, col]] = data[[row, col * 3 + 1]];
            blue[[row, col]] = data[[row, col * 3 + 2]];
        }
    }

    ColorFrame {
        red: Frame::new(red, bit_depth),
        green: Frame::new(green, bit_depth),
        blue: Frame::new(blue, bit_depth),
    }
}

/// Merge separate R, G, B frames into an interleaved RGB Array2.
pub fn merge_rgb(color: &ColorFrame) -> Array2<f32> {
    let (h, w) = color.red.data.dim();
    let mut data = Array2::<f32>::zeros((h, w * 3));

    for row in 0..h {
        for col in 0..w {
            data[[row, col * 3]] = color.red.data[[row, col]];
            data[[row, col * 3 + 1]] = color.green.data[[row, col]];
            data[[row, col * 3 + 2]] = color.blue.data[[row, col]];
        }
    }

    data
}

/// Apply a processing function to each channel of a color frame independently.
pub fn process_color<F>(color: &ColorFrame, mut process_fn: F) -> ColorFrame
where
    F: FnMut(&Frame) -> Frame,
{
    ColorFrame {
        red: process_fn(&color.red),
        green: process_fn(&color.green),
        blue: process_fn(&color.blue),
    }
}

/// Create a ColorFrame from three separate mono frames.
pub fn from_channels(red: Frame, green: Frame, blue: Frame) -> ColorFrame {
    ColorFrame { red, green, blue }
}
