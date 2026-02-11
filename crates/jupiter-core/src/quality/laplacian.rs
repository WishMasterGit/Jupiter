use ndarray::Array2;
use rayon::prelude::*;

use crate::frame::{Frame, QualityScore};

/// Compute Laplacian variance of a frame — higher means sharper.
///
/// Convolves with the 3x3 Laplacian kernel:
///   0  1  0
///   1 -4  1
///   0  1  0
/// Then returns the variance of the result.
pub fn laplacian_variance(frame: &Frame) -> f64 {
    laplacian_variance_array(&frame.data)
}

pub fn laplacian_variance_array(data: &Array2<f32>) -> f64 {
    let (h, w) = data.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let count = ((h - 2) * (w - 2)) as f64;

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let lap = -4.0 * data[[row, col]] as f64
                + data[[row - 1, col]] as f64
                + data[[row + 1, col]] as f64
                + data[[row, col - 1]] as f64
                + data[[row, col + 1]] as f64;
            sum += lap;
            sum_sq += lap * lap;
        }
    }

    let mean = sum / count;
    sum_sq / count - mean * mean
}

/// Score all frames and return (index, QualityScore) sorted by quality descending.
pub fn rank_frames(frames: &[Frame]) -> Vec<(usize, QualityScore)> {
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let lv = laplacian_variance(f);
            (
                i,
                QualityScore {
                    laplacian_variance: lv,
                    composite: lv,
                },
            )
        })
        .collect();

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    scores
}

/// Select indices of the best `percentage` (0.0..1.0) of frames.
pub fn select_best(frames: &[Frame], percentage: f32) -> Vec<usize> {
    let ranked = rank_frames(frames);
    let keep = ((frames.len() as f32 * percentage.clamp(0.0, 1.0)).ceil()) as usize;
    ranked.into_iter().take(keep).map(|(i, _)| i).collect()
}
