//! Gradient cross-correlation alignment.
//!
//! Applies Sobel gradient magnitude preprocessing to both images before
//! running standard FFT phase correlation. This emphasizes edges and detail,
//! making alignment more robust in noisy conditions.

use ndarray::Array2;

use crate::compute::ComputeBackend;
use crate::error::Result;
use crate::frame::AlignmentOffset;
use crate::quality::gradient::gradient_magnitude_array;

use super::phase_correlation::{compute_offset_array, compute_offset_gpu};

/// Compute alignment offset using gradient cross-correlation.
///
/// Applies Sobel gradient magnitude filter to both images, then runs
/// standard phase correlation on the gradient images.
pub fn compute_offset_gradient(
    reference: &Array2<f32>,
    target: &Array2<f32>,
    backend: &dyn ComputeBackend,
) -> Result<AlignmentOffset> {
    let ref_grad = gradient_magnitude_array(reference);
    let tgt_grad = gradient_magnitude_array(target);

    if backend.is_gpu() {
        compute_offset_gpu(&ref_grad, &tgt_grad, backend)
    } else {
        compute_offset_array(&ref_grad, &tgt_grad)
    }
}
