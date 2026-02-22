use ndarray::Array2;

use crate::error::{JupiterError, Result};
use crate::frame::Frame;

/// Stack frames by computing the mean at each pixel.
pub fn mean_stack(frames: &[Frame]) -> Result<Frame> {
    mean_stack_with_progress(frames, |_| {})
}

/// Stack frames by computing the mean at each pixel, with per-frame progress.
///
/// `on_progress` is called with the cumulative number of frames accumulated.
pub fn mean_stack_with_progress(frames: &[Frame], on_progress: impl Fn(usize)) -> Result<Frame> {
    if frames.is_empty() {
        return Err(JupiterError::EmptySequence);
    }

    let (h, w) = frames[0].data.dim();
    let n = frames.len() as f32;

    let mut sum = Array2::<f32>::zeros((h, w));

    for (i, frame) in frames.iter().enumerate() {
        sum += &frame.data;
        on_progress(i + 1);
    }

    sum /= n;

    Ok(Frame::new(sum, frames[0].original_bit_depth))
}

/// Streaming mean stacker that accumulates one frame at a time.
///
/// Memory usage is O(h*w) regardless of frame count — only the running sum
/// and a count are stored. Frames can be dropped immediately after `add()`.
pub struct StreamingMeanStacker {
    sum: Array2<f32>,
    count: usize,
    bit_depth: u8,
}

impl StreamingMeanStacker {
    pub fn new(height: usize, width: usize, bit_depth: u8) -> Self {
        Self {
            sum: Array2::zeros((height, width)),
            count: 0,
            bit_depth,
        }
    }

    /// Add one frame to the running sum.
    pub fn add(&mut self, frame: &Frame) {
        self.sum += &frame.data;
        self.count += 1;
    }

    /// Produce the final mean-stacked frame.
    pub fn finalize(mut self) -> Result<Frame> {
        if self.count == 0 {
            return Err(JupiterError::EmptySequence);
        }
        self.sum /= self.count as f32;
        Ok(Frame::new(self.sum, self.bit_depth))
    }
}
