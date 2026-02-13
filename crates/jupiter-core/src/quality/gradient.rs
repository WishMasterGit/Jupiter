use ndarray::Array2;
use rayon::prelude::*;

use crate::consts::STREAMING_BATCH_SIZE;
use crate::error::Result;
use crate::frame::{Frame, QualityScore};
use crate::io::ser::SerReader;

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

/// Score all frames using gradient metric by reading in batches from the SER reader.
///
/// Each batch is decoded, scored in parallel, then dropped. This avoids holding
/// all N frames in memory simultaneously.
///
/// Returns `(index, QualityScore)` sorted by quality descending.
pub fn rank_frames_gradient_streaming(reader: &SerReader) -> Result<Vec<(usize, QualityScore)>> {
    let total = reader.frame_count();
    let mut scores: Vec<(usize, QualityScore)> = Vec::with_capacity(total);

    for batch_start in (0..total).step_by(STREAMING_BATCH_SIZE) {
        let batch_end = (batch_start + STREAMING_BATCH_SIZE).min(total);
        let batch: Vec<(usize, Frame)> = (batch_start..batch_end)
            .map(|i| Ok((i, reader.read_frame(i)?)))
            .collect::<Result<_>>()?;

        let batch_scores: Vec<(usize, QualityScore)> = batch
            .par_iter()
            .map(|(i, frame)| {
                let gs = gradient_score(frame);
                (
                    *i,
                    QualityScore {
                        laplacian_variance: 0.0,
                        composite: gs,
                    },
                )
            })
            .collect();

        scores.extend(batch_scores);
    }

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    Ok(scores)
}
