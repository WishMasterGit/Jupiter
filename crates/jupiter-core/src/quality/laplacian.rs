use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use rayon::prelude::*;

use crate::color::debayer::{luminance, DebayerMethod};
use crate::consts::STREAMING_BATCH_SIZE;
use crate::error::Result;
use crate::frame::{ColorMode, Frame, QualityScore};
use crate::io::ser::SerReader;

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

/// Score all frames with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` as each frame is scored.
pub fn rank_frames_with_progress(
    frames: &[Frame],
    on_progress: impl Fn(usize) + Send + Sync,
) -> Vec<(usize, QualityScore)> {
    let done = AtomicUsize::new(0);
    let mut scores: Vec<(usize, QualityScore)> = frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            let lv = laplacian_variance(f);
            let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
            on_progress(completed);
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

/// Score all frames by reading them in batches from the SER reader.
///
/// Each batch of `STREAMING_BATCH_SIZE` frames is decoded, scored in parallel,
/// then dropped before the next batch is loaded. This avoids holding all N
/// frames in memory simultaneously.
///
/// Returns `(index, QualityScore)` sorted by quality descending.
pub fn rank_frames_streaming(reader: &SerReader) -> Result<Vec<(usize, QualityScore)>> {
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
                let lv = laplacian_variance(frame);
                (
                    *i,
                    QualityScore {
                        laplacian_variance: lv,
                        composite: lv,
                    },
                )
            })
            .collect();

        scores.extend(batch_scores);
        // batch dropped here — memory freed
    }

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    Ok(scores)
}

/// Score all frames streaming with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` after each batch is scored.
pub fn rank_frames_streaming_with_progress(
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
                let lv = laplacian_variance(frame);
                (
                    *i,
                    QualityScore {
                        laplacian_variance: lv,
                        composite: lv,
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

/// Score color frames by reading them in batches from the SER reader.
///
/// For each batch: read raw frames, debayer (or split RGB), convert to
/// luminance, score in parallel, then drop the batch.
///
/// Returns `(index, QualityScore)` sorted by quality descending.
pub fn rank_frames_color_streaming(
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
                let lv = laplacian_variance(frame);
                (
                    *i,
                    QualityScore {
                        laplacian_variance: lv,
                        composite: lv,
                    },
                )
            })
            .collect();

        scores.extend(batch_scores);
    }

    scores.sort_by(|a, b| b.1.composite.partial_cmp(&a.1.composite).unwrap());
    Ok(scores)
}

/// Score color frames streaming with per-frame progress reporting.
///
/// Calls `on_progress(items_done)` after each batch is scored.
pub fn rank_frames_color_streaming_with_progress(
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
                let lv = laplacian_variance(frame);
                (
                    *i,
                    QualityScore {
                        laplacian_variance: lv,
                        composite: lv,
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
