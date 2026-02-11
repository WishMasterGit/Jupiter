use ndarray::Array2;
use rayon::prelude::*;

use crate::frame::{Frame, QualityScore};

/// Compute gradient magnitude quality score on raw array data.
pub fn gradient_score_array(data: &Array2<f32>) -> f64 {
    let (h, w) = data.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let count = ((h - 2) * (w - 2)) as f64;

    for row in 1..h - 1 {
        for col in 1..w - 1 {
            let gx = -data[[row - 1, col - 1]] as f64
                + data[[row - 1, col + 1]] as f64
                - 2.0 * data[[row, col - 1]] as f64
                + 2.0 * data[[row, col + 1]] as f64
                - data[[row + 1, col - 1]] as f64
                + data[[row + 1, col + 1]] as f64;

            let gy = -data[[row - 1, col - 1]] as f64
                - 2.0 * data[[row - 1, col]] as f64
                - data[[row - 1, col + 1]] as f64
                + data[[row + 1, col - 1]] as f64
                + 2.0 * data[[row + 1, col]] as f64
                + data[[row + 1, col + 1]] as f64;

            sum += (gx * gx + gy * gy).sqrt();
        }
    }

    sum / count
}

/// Compute gradient magnitude quality score using Sobel operator.
///
/// Sobel kernels:
///   Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
///   Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
///
/// Score = mean of sqrt(Gx² + Gy²). Higher = sharper.
pub fn gradient_score(frame: &Frame) -> f64 {
    gradient_score_array(&frame.data)
}

/// Score all frames using gradient metric and return sorted by quality descending.
pub fn rank_frames_gradient(frames: &[Frame]) -> Vec<(usize, QualityScore)> {
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let gs = gradient_score(f);
            (
                i,
                QualityScore {
                    laplacian_variance: 0.0,
                    composite: gs,
                },
            )
        })
        .collect();

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    scores
}
