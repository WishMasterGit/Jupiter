//! Coarse-to-fine Gaussian pyramid alignment.
//!
//! Builds a multi-level Gaussian pyramid and performs phase correlation
//! at each level from coarsest to finest. This handles large displacements
//! (> 50% of the image size) that exceed the FFT wrap-around limit of
//! standard phase correlation.

use ndarray::Array2;

use crate::compute::ComputeBackend;
use crate::consts::PYRAMID_BLUR_SIGMA;
use crate::error::Result;
use crate::filters::gaussian_blur::gaussian_blur_array;
use crate::frame::AlignmentOffset;
use crate::pipeline::config::PyramidConfig;

use super::phase_correlation::{compute_offset_array, compute_offset_gpu, shift_array};

/// Compute alignment offset using coarse-to-fine pyramid alignment.
///
/// Builds a Gaussian pyramid for each image, then iteratively refines
/// the offset from the coarsest level to the original resolution.
pub fn compute_offset_pyramid(
    reference: &Array2<f32>,
    target: &Array2<f32>,
    config: &PyramidConfig,
    backend: &dyn ComputeBackend,
) -> Result<AlignmentOffset> {
    let levels = config.levels;

    let ref_pyramid = build_pyramid(reference, levels);
    let tgt_pyramid = build_pyramid(target, levels);

    let mut offset = AlignmentOffset::default();

    // Iterate from coarsest (last) to finest (first = original)
    for level in (0..=levels).rev() {
        let ref_level = &ref_pyramid[level];
        let tgt_level = &tgt_pyramid[level];

        // Scale offset from previous (coarser) level
        if level < levels {
            offset.dx *= 2.0;
            offset.dy *= 2.0;
        }

        // Shift target by accumulated offset, then correlate the residual
        let shifted_target = shift_array(tgt_level, &offset);

        let residual = if backend.is_gpu() {
            compute_offset_gpu(ref_level, &shifted_target, backend)?
        } else {
            compute_offset_array(ref_level, &shifted_target)?
        };

        offset.dx += residual.dx;
        offset.dy += residual.dy;
    }

    Ok(offset)
}

/// Build a Gaussian pyramid with `levels` downsampled levels.
///
/// Returns a vector of `levels + 1` arrays, where index 0 is the original
/// and index `levels` is the coarsest.
fn build_pyramid(data: &Array2<f32>, levels: usize) -> Vec<Array2<f32>> {
    let mut pyramid = Vec::with_capacity(levels + 1);
    pyramid.push(data.clone());

    let mut current = data.clone();
    for _ in 0..levels {
        let blurred = gaussian_blur_array(&current, PYRAMID_BLUR_SIGMA);
        let downsampled = downsample_2x(&blurred);
        current = downsampled;
        pyramid.push(current.clone());
    }

    pyramid
}

/// Downsample an image by 2x by taking every other pixel.
fn downsample_2x(data: &Array2<f32>) -> Array2<f32> {
    let (h, w) = data.dim();
    let new_h = (h + 1) / 2;
    let new_w = (w + 1) / 2;
    let mut result = Array2::<f32>::zeros((new_h, new_w));

    for r in 0..new_h {
        for c in 0..new_w {
            result[[r, c]] = data[[r * 2, c * 2]];
        }
    }

    result
}
