use ndarray::Array2;

use crate::error::{JupiterError, Result};
use crate::frame::Frame;

/// Stack frames by computing the mean at each pixel.
pub fn mean_stack(frames: &[Frame]) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len() as f32;

    let mut sum = Array2::<f32>::zeros((h, w));

    for frame in frames {
        sum += &frame.data;
    }

    sum /= n;

    Ok(Frame::new(sum, frames[0].original_bit_depth))
}
