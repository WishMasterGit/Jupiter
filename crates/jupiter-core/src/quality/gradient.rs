use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use rayon::prelude::*;

use crate::color::debayer::{luminance, DebayerMethod};
use crate::consts::STREAMING_BATCH_SIZE;
use crate::error::Result;
use crate::frame::{ColorMode, Frame, QualityScore};
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

/// Score all frames using gradient metric with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` as each frame is scored.
pub fn rank_frames_gradient_with_progress(
    frames: &[Frame],
    on_progress: impl Fn(usize) + Send + Sync,
) -> Vec<(usize, QualityScore)> {
    let done = AtomicUsize::new(0);
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let gs = gradient_score(f);
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(completed);
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

/// Score all frames using gradient metric streaming with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` after each batch is scored.
pub fn rank_frames_gradient_streaming_with_progress(
    reader: &SerReader,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
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
        on_progress(scores.len());
    }

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    Ok(scores)
}

/// Score color frames using gradient metric by reading in batches from the SER reader.
///
/// Each batch is read, debayered/split, converted to luminance, scored in parallel,
/// then dropped.
///
/// Returns `(index, QualityScore)` sorted by quality descending.
pub fn rank_frames_gradient_color_streaming(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
) -> Result<Vec<(usize, QualityScore)>> {
    let total = reader.frame_count();
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);
    let mut scores: Vec<(usize, QualityScore)> = Vec::with_capacity(total);

    for batch_start in (0..total).step_by(STREAMING_BATCH_SIZE) {
        let batch_end = (batch_start + STREAMING_BATCH_SIZE).min(total);
        let batch: Vec<(usize, Frame)> = (batch_start..batch_end)
            .map(|i| {
                let color_frame = if is_rgb_bgr {
                    reader.read_frame_rgb(i)?
                } else {
                    reader.read_frame_color(i, debayer_method)?
                };
                Ok((i, luminance(&color_frame)))
            })
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

/// Score color frames using gradient metric streaming with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` after each batch is scored.
pub fn rank_frames_gradient_color_streaming_with_progress(
    reader: &SerReader,
    color_mode: &ColorMode,
    debayer_method: &DebayerMethod,
    on_progress: impl Fn(usize),
) -> Result<Vec<(usize, QualityScore)>> {
    let total = reader.frame_count();
    let is_rgb_bgr = matches!(color_mode, ColorMode::RGB | ColorMode::BGR);
    let mut scores: Vec<(usize, QualityScore)> = Vec::with_capacity(total);

    for batch_start in (0..total).step_by(STREAMING_BATCH_SIZE) {
        let batch_end = (batch_start + STREAMING_BATCH_SIZE).min(total);
        let batch: Vec<(usize, Frame)> = (batch_start..batch_end)
            .map(|i| {
                let color_frame = if is_rgb_bgr {
                    reader.read_frame_rgb(i)?
                } else {
                    reader.read_frame_color(i, debayer_method)?
                };
                Ok((i, luminance(&color_frame)))
            })
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
        on_progress(scores.len());
    }

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    Ok(scores)
}
