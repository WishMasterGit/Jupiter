//! Centroid (center-of-gravity) alignment.
//!
//! Computes the intensity-weighted center of mass of each image and returns
//! the difference as the alignment offset. Very fast (O(n)), naturally sub-pixel,
//! and works well for bright planetary disks.

use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::PARALLEL_PIXEL_THRESHOLD;
use crate::error::Result;
use crate::frame::AlignmentOffset;
use crate::pipeline::config::CentroidConfig;

/// Compute alignment offset between two images using intensity-weighted centroid.
pub fn compute_offset_centroid(
    reference: &Array2<f32>,
    target: &Array2<f32>,
    config: &CentroidConfig,
) -> Result<AlignmentOffset> {
    let (ref_y, ref_x) = compute_centroid(reference, config.threshold);
    let (tgt_y, tgt_x) = compute_centroid(target, config.threshold);

    Ok(AlignmentOffset {
        dx: tgt_x - ref_x,
        dy: tgt_y - ref_y,
    })
}

/// Compute the intensity-weighted centroid of an image.
///
/// Pixels below `threshold * max_intensity` are excluded.
/// Returns `(center_row, center_col)` in pixel coordinates.
fn compute_centroid(data: &Array2<f32>, threshold: f32) -> (f64, f64) {
    let (h, w) = data.dim();
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if max_val <= 0.0 {
        // All-black or empty image — return geometric center.
        return (h as f64 / 2.0, w as f64 / 2.0);
    }

    let cutoff = threshold * max_val;

    if h * w >= PARALLEL_PIXEL_THRESHOLD {
        compute_centroid_parallel(data, cutoff, h, w)
    } else {
        compute_centroid_sequential(data, cutoff, h, w)
    }
}

/// Row-parallel centroid summation using Rayon.
fn compute_centroid_parallel(data: &Array2<f32>, cutoff: f32, h: usize, w: usize) -> (f64, f64) {
    let row_sums: Vec<(f64, f64, f64)> = (0..h)
        .into_par_iter()
        .map(|row| {
            let mut sum_r = 0.0f64;
            let mut sum_c = 0.0f64;
            let mut sum_w = 0.0f64;
            for col in 0..w {
                let val = data[[row, col]];
                if val > cutoff {
                    let weight = val as f64;
                    sum_r += row as f64 * weight;
                    sum_c += col as f64 * weight;
                    sum_w += weight;
                }
            }
            (sum_r, sum_c, sum_w)
        })
        .collect();

    let (total_r, total_c, total_w) = row_sums
        .into_iter()
        .fold((0.0, 0.0, 0.0), |(ar, ac, aw), (r, c, w)| {
            (ar + r, ac + c, aw + w)
        });

    if total_w > 0.0 {
        (total_r / total_w, total_c / total_w)
    } else {
        (h as f64 / 2.0, w as f64 / 2.0)
    }
}

/// Sequential centroid summation using nested loops.
fn compute_centroid_sequential(
    data: &Array2<f32>,
    cutoff: f32,
    h: usize,
    w: usize,
) -> (f64, f64) {
    let mut sum_r = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut sum_w = 0.0f64;

    for row in 0..h {
        for col in 0..w {
            let val = data[[row, col]];
            if val > cutoff {
                let weight = val as f64;
                sum_r += row as f64 * weight;
                sum_c += col as f64 * weight;
                sum_w += weight;
            }
        }
    }

    if sum_w > 0.0 {
        (sum_r / sum_w, sum_c / sum_w)
    } else {
        (h as f64 / 2.0, w as f64 / 2.0)
    }
}
