use ndarray::Array2;

use crate::error::{JupiterError, Result};
use crate::frame::Frame;

/// Stack frames by computing the median at each pixel position.
///
/// Uses `select_nth_unstable` for O(n) median without full sort.
pub fn median_stack(frames: &[Frame]) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len();
    let mut result = Array2::<f32>::zeros((h, w));

    let mut pixel_values = vec![0.0f32; n];

    for row in 0..h {
        for col in 0..w {
            for (i, frame) in frames.iter().enumerate() {
                pixel_values[i] = frame.data[[row, col]];
            }

            result[[row, col]] = if n == 1 {
                pixel_values[0]
            } else if n % 2 == 1 {
                let mid = n / 2;
                *pixel_values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap()).1
            } else {
                let mid = n / 2;
                let (_, upper, _) =
                    pixel_values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
                let upper_val = *upper;
                // For even count, average the two middle values
                let lower_val = pixel_values[..mid]
                    .iter()
                    .copied()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                (lower_val + upper_val) / 2.0
            };
        }
    }

    Ok(Frame::new(result, frames[0].original_bit_depth))
}
