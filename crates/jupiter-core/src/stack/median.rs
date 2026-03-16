use crate::error::{JupiterError, Result};
use crate::frame::Frame;
use crate::utils::{compute_median, par_or_seq_rows};

/// Stack frames by computing the median at each pixel position.
///
/// Uses `select_nth_unstable` for O(n) median without full sort.
/// Parallelizes at the row level for images >= 256x256.
pub fn median_stack(frames: &[Frame]) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len();

    let result = par_or_seq_rows(h, w, |row| {
        let mut pixel_values = vec![0.0f32; n];
        let mut row_result = vec![0.0f32; w];
        for (col, result) in row_result.iter_mut().enumerate() {
            for (i, frame) in frames.iter().enumerate() {
                pixel_values[i] = frame.data[[row, col]];
            }
            *result = compute_median(&mut pixel_values, n);
        }
        row_result
    });

    Ok(Frame::new(result, frames[0].original_bit_depth))
}
